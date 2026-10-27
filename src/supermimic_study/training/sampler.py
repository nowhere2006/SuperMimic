"""Return-based one-second reset bins (Appendix D.1).

EMA rate and accumulation of historical softmax weights are explicit engineering
choices because the paper does not specify those estimators.
"""
from __future__ import annotations

import torch


class ReturnBinSampler:
    def __init__(self, frame_counts, *, clip_ids=None, fps=50, gamma=.99,
                 maximum_step_reward=.1, ema_rate=.1, device="cpu"):
        self.device = torch.device(device)
        self.fps, self.gamma, self.ema_rate = fps, gamma, ema_rate
        self.gmax = maximum_step_reward/(1-gamma)
        clips = range(len(frame_counts)) if clip_ids is None else clip_ids
        bins = [(c, start, min(start+fps, frame_counts[c]-1)) for c in clips
                for start in range(0, frame_counts[c]-1, fps)]
        if not bins:
            raise ValueError("no reference reset bins with a successor frame")
        self.bins = torch.tensor(bins, dtype=torch.long, device=device)
        self.returns = torch.zeros(len(bins), device=device)
        self.visits = torch.zeros(len(bins), dtype=torch.long, device=device)
        self.historical = torch.ones(len(bins), device=device)/len(bins)
        self._lookup = {(int(c), int(s)//fps): i for i, (c, s, _) in enumerate(bins)}

    def probabilities(self):
        hard = torch.softmax(-25*self.returns/self.gmax, dim=0)
        return .5/len(hard) + .4*hard + .1*self.historical/self.historical.sum()

    def sample(self, count):
        index = torch.multinomial(self.probabilities(), count, replacement=True)
        selected = self.bins[index]
        frame = selected[:, 1]+(torch.rand(count, device=self.device)*
                                (selected[:, 2]-selected[:, 1])).long()
        self.historical += torch.softmax(-25*self.returns/self.gmax, 0)
        return selected[:, 0].clone(), frame, index

    def update_episode(self, clips, frames, rewards):
        """Discounted suffix return at every visited frame; aggregate once per bin."""
        if not (len(clips) == len(frames) == len(rewards)):
            raise ValueError("episode log lengths differ")
        suffix, by_bin = 0., {}
        for c, f, r in reversed(list(zip(clips, frames, rewards))):
            suffix = float(r)+self.gamma*suffix
            key = self._lookup[(int(c), int(f)//self.fps)]
            by_bin.setdefault(key, []).append(suffix)
        for key, values in by_bin.items():
            mean = sum(values)/len(values)
            self.returns[key] = (1-self.ema_rate)*self.returns[key]+self.ema_rate*mean
            self.visits[key] += 1

    def state_dict(self):
        return {"bins": self.bins.clone(), "returns": self.returns.clone(),
                "visits": self.visits.clone(), "historical": self.historical.clone()}

    def load_state_dict(self, state):
        if set(state) != {"bins","returns","visits","historical"}:
            raise ValueError("sampler checkpoint fields differ")
        bins = state["bins"].to(self.device)
        if not torch.equal(bins,self.bins):
            raise ValueError("sampler reset bins differ from checkpoint")
        for name in ("returns","visits","historical"):
            incoming = state[name].to(self.device)
            current = getattr(self,name)
            if incoming.shape != current.shape or not torch.isfinite(incoming).all():
                raise ValueError(f"invalid sampler checkpoint {name}")
            if name in ("visits","historical") and (incoming < 0).any():
                raise ValueError(f"negative sampler checkpoint {name}")
        if state["historical"].sum() <= 0:
            raise ValueError("empty historical sampler distribution")
        for name in ("returns","visits","historical"):
            getattr(self,name).copy_(state[name].to(self.device))
