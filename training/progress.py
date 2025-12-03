class ProgressMeter:
    """
    Tracks training progress in terms of processed tokens, and coordinates
    evaluation and checkpoint scheduling based on configurable intervals.

    API preserved from inline definition in train.py.
    """

    def __init__(
        self,
        target_tokens: int,
        *,
        eval_every_tokens: int | None,
        checkpoint_per_n_tokens: int | None,
        checkpoint_warmup_tokens: int = 0,
    ) -> None:
        """
        Initializes the object with a token processing schedule and optional evaluation
        and checkpointing configurations.

        :param target_tokens: The total number of tokens to be processed.
        :param eval_every_tokens: Optional. The number of tokens after which an evaluation is scheduled.
        :param checkpoint_per_n_tokens: Optional. The number of tokens after which a checkpoint
            is scheduled. If set to 0, checkpointing will occur immediately after warmup.
        :param checkpoint_warmup_tokens: Optional. The number of tokens to process during
            the warmup phase before enabling checkpointing.
        """
        self.target_tokens = int(target_tokens)
        self.tokens_processed = 0
        self.eval_every_tokens = int(eval_every_tokens) if eval_every_tokens else None
        self.next_eval_at = 0
        # checkpoint schedule: allow 0 meaning "after warmup, then once per update"
        self.checkpoint_per_n_tokens = int(checkpoint_per_n_tokens) if checkpoint_per_n_tokens is not None and checkpoint_per_n_tokens >= 0 else None
        self.checkpoint_warmup_tokens = int(checkpoint_warmup_tokens or 0)
        if self.checkpoint_per_n_tokens is None:
            self.next_checkpoint_at = None
        else:
            # initial schedule: at warmup + interval (interval may be 0)
            self.next_checkpoint_at = self.checkpoint_warmup_tokens + self.checkpoint_per_n_tokens
            # if interval is 0, ensure we at least schedule on or after warmup boundary
            if self.checkpoint_per_n_tokens == 0 and self.next_checkpoint_at < self.checkpoint_warmup_tokens:
                self.next_checkpoint_at = self.checkpoint_warmup_tokens

    @property
    def s(self) -> float:
        if self.target_tokens <= 0:
            return 0.0
        return min(self.tokens_processed / self.target_tokens, 1.0)

    def update(self, step_tokens: int) -> None:
        self.tokens_processed += int(step_tokens)

    def should_eval(self) -> bool:
        if self.eval_every_tokens is None or self.next_eval_at is None:
            return False
        return self.tokens_processed >= self.next_eval_at

    def mark_eval_done(self) -> None:
        if self.eval_every_tokens is not None and self.next_eval_at is not None:
            while self.tokens_processed >= self.next_eval_at:
                self.next_eval_at += self.eval_every_tokens

    def should_checkpoint(self) -> bool:
        if self.checkpoint_per_n_tokens is None or self.next_checkpoint_at is None:
            return False
        if self.tokens_processed < self.checkpoint_warmup_tokens:
            return False
        return self.tokens_processed >= self.next_checkpoint_at

    def mark_checkpoint_done(self) -> None:
        if self.checkpoint_per_n_tokens is not None and self.next_checkpoint_at is not None:
            if self.checkpoint_per_n_tokens == 0:
                # schedule next checkpoint at least one token after current to avoid retriggering at same count
                self.next_checkpoint_at = self.tokens_processed + 1
            else:
                while self.tokens_processed >= self.next_checkpoint_at:
                    self.next_checkpoint_at += self.checkpoint_per_n_tokens

    def state_dict(self) -> dict:
        return {
            "tokens_processed": self.tokens_processed,
            "next_eval_at": self.next_eval_at,
            "next_checkpoint_at": self.next_checkpoint_at,
            "target_tokens": self.target_tokens,
        }

    def load_state_dict(self, d: dict) -> None:
        self.tokens_processed = int(d.get("tokens_processed", 0))
        self.next_eval_at = d.get("next_eval_at", self.eval_every_tokens)
        self.next_checkpoint_at = d.get("next_checkpoint_at", self.next_checkpoint_at)
