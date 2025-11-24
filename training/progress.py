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
        Initializes a tracking object to manage token processing and evaluation or checkpoint scheduling.

        This class facilitates the process of managing checkpoints and evaluations during token
        processing. It ensures that these actions conform to a set of specified intervals and
        warmup limits.

        Parameters:
            target_tokens (int): The total number of tokens targeted to process.
            eval_every_tokens (int | None): Optional interval in tokens after which evaluation
                                              is triggered. If None, no evaluation is scheduled.
            checkpoint_per_n_tokens (int | None): Optional interval in tokens to trigger a
                                                  checkpoint. If None or set to a non-positive
                                                  number, no scheduled checkpoints occur.
            checkpoint_warmup_tokens (int): Number of initial tokens to process before activating
                                             checkpoint scheduling. Defaults to 0.

        Attributes:
            target_tokens (int): The total number of tokens targeted to process.
            tokens_processed (int): The count of tokens processed so far, initialized to 0.
            eval_every_tokens (int | None): The interval in tokens after which evaluation is
                                             triggered, if specified.
            next_eval_at (int | None): The next token count at which an evaluation is scheduled,
                                       if applicable.
            checkpoint_per_n_tokens (int | None): The interval in tokens to trigger a checkpoint,
                                                  if applicable.
            checkpoint_warmup_tokens (int): The number of tokens to process before enabling the
                                             checkpoint mechanism.
            next_checkpoint_at (int | None): The next token count at which a checkpoint is scheduled
                                              after accounting for warmup, if applicable.
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
