from typing import Callable, Optional, Tuple

import torch
from tqdm import tqdm




# algorithm
    def _sample_sequence_in_time_dimension(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        return_all: bool = False,
        number_of_chunks: int = 1,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        x_shape = self.hparams.x_shape
        padding = self.max_tokens - length

        scheduling_matrix = self._generate_scheduling_matrix(
            length,
            0,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)

        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        # Prepare the record list if we want all intermediate steps
        record = [] if return_all else None

        # Initial random noise
        xs_pred = torch.randn(
            (batch_size, length, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.hparams.clip_noise, self.hparams.clip_noise)
        sliding_window_context_length = 0 # the the overlap between the context and the generated frames
        h = self.max_tokens - sliding_window_context_length
        l = sliding_window_context_length + h



        # Create a single progress bar if none is given
        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling with DFoT",
                leave=False,
            )

        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels_full = scheduling_matrix[m]     # shape: [b, t]
            to_noise_levels_full = scheduling_matrix[m + 1]   # shape: [b, t]
            number_of_generations = number_of_chunks

            if return_all:
                record.append(xs_pred.clone())

            for i in range(number_of_generations):
                # in sliding window fashion
                idx = self.max_tokens - sliding_window_context_length if i > 0 else self.max_tokens
                from_noise_levels = from_noise_levels_full[:, i*idx:(i+1)*idx]
                to_noise_levels   = to_noise_levels_full[:, i*idx:(i+1)*idx]
                context_in = context[:, i*idx:(i+1)*idx]
                context_mask_in = context_mask[:, i*idx:(i+1)*idx]
                xs_pred_in = xs_pred[:, i*idx:(i+1)*idx]
                conditions_in = conditions[:, i*idx:(i+1)*idx] if conditions is not None else None
                
                # If context is None, create a dummy context + mask, also if we are not in the first iteration
                if context is None:
                    context_in = torch.zeros_like(xs_pred)
                    context_mask_in = torch.zeros(
                        (batch_size, self.max_tokens),
                        dtype=torch.long,
                        device=self.device
                    )
                    # context mask is 1 for sliding_window_context_length frames and 0 for the rest
                context_mask_in = torch.cat(
                    [
                        torch.ones((batch_size, sliding_window_context_length), dtype=torch.long, device=self.device),
                        torch.zeros((batch_size, h), dtype=torch.long, device=self.device),
                    ],
                    dim=1,
                )

                xs_pred_prev_in = xs_pred_in.clone()
                # If we do NOT concatenate context into the diffusion model's channels,
                # we literally replace the context frames in xs_pred with the given context
                if not self.hparams.cat_context_in_c_dim or sliding_window_context_length > 0:
                    xs_pred_in = torch.where(self._extend_x_dim(context_mask_in) >= 1, context_in, xs_pred_in)

                # Make a backup for context tokens so we can revert them after diffusion

                # If we DO concatenate context in c-dim, cat them here
                if self.hparams.cat_context_in_c_dim:
                    xs_pred_in = torch.cat([xs_pred_in, context_in], dim=2)  # depends on your exact shape

                # Single diffusion step from one noise level to another
                xs_pred_in, aux_output = self.diffusion_model.sample_step(
                    xs_pred_in,
                    from_noise_levels,
                    to_noise_levels,
                    self._process_conditions(conditions_in, from_noise_levels),
                    #conditions_mask=None,
                    guidance_fn=None,
                )

                # If we concatenated context channels, revert the shape
                if self.hparams.cat_context_in_c_dim:
                    # removing last channels
                    xs_pred_in = xs_pred_in[:, :, : x_shape[0]]

                # Revert context tokens to their original values
                xs_pred_in = torch.where(
                    self._extend_x_dim(context_mask_in) == 0, xs_pred_in, xs_pred_prev_in
                )
                xs_pred[:, i*idx:(i+1)*idx] = xs_pred_in

        return xs_pred, record



def _predict_sequence(
    self,
    context: torch.Tensor,   # (batch_size, self.n_context_tokens, *x_shape) just gt frames
    length: Optional[int] = None, # self.n_tokens frames  (determined over xs_pred)
    conditions: Optional[torch.Tensor] = None,
    guidance_fn: Optional[Callable] = None,
    reconstruction_guidance: float = 0.0,
    sliding_context_len: Optional[int] = None, # self.hparams.sliding_context_len 
    return_all: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Predict a sequence given initial context frames, possibly in a
    rolling/sliding window approach if length > self.max_tokens.
    """
    if sliding_context_len == -1:
        sliding_context_len = self.max_tokens - 1

    batch_size, gt_len, *_ = context.shape
    
    curr_token = gt_len
    xs_pred = context
    record = None

    # How many total diffusion steps for the entire rolling process?
    # The factor accounts for how many “chunks” we will sample
    number_of_chunks = 1 + max(0, (length - sliding_context_len - 1)) // max(1, (self.max_tokens - sliding_context_len))
    total_passes = self.hparams.sampling_timesteps * number_of_chunks

    pbar = tqdm(
        total=total_passes,
        initial=0,
        desc="Predicting (vanilla diffusion)",
        leave=False,
    )

    # Rolling from left to right until we generate 'length' frames:
    iteration = 0
    while curr_token < length:
        # If storing all steps, forbid sliding windows for simplicity
        if return_all:
            raise ValueError("return_all is not supported with sliding window.")

        # Decide how many frames of context vs. how many frames to generate:
        if sliding_context_len == 0 and iteration == 0:
            c = curr_token
        else: 
            c = min(sliding_context_len, curr_token)


        h =  self.max_tokens - c
        l = c + h # total length fed to the model

        if self.hparams.generate_in_noise_dim:
            l = (self.n_tokens // self.max_tokens) * self.max_tokens  # total frames to be generated
            h = l - c
            if h < 0:
                raise ValueError("Context length is larger than the total number of tokens.")
            

        # Prepare next chunk (context chunk + blank frames)
        pad = torch.zeros((batch_size, h, *self.hparams.x_shape), device=self.device)

        context_mask = torch.cat(
            [
                # Mark c tokens as “context” (1)
                torch.ones((batch_size, c), dtype=torch.long, device=self.device),
                # Mark h tokens as “to be generated” (0)
                torch.zeros((batch_size, h), dtype=torch.long, device=self.device),
            ],
            dim=1,
        )

        context_chunk = torch.cat(
            [xs_pred[:, -c:] if c > 0 else torch.empty_like(pad[:, :0]), pad], dim=1
        )

        if conditions is not None and self.n_tokens > self.max_tokens:  #self.hparams.use_causal_mask:
            cond_chunk = conditions[:, curr_token - c : curr_token - c + l]
        else:
            cond_chunk = conditions

        # -----------------------------------

        scheduling_matrix = self._generate_scheduling_matrix(
            l,
            0,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)

        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        record = [] if return_all else None


        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]

            if self.hparams.generate_in_noise_dim:
                number_of_generations = number_of_chunks
            else:
                number_of_generations = 1
            

            for i in range(number_of_generations):
                if self.hparams.generate_in_noise_dim:
                    from_noise_levels = from_noise_levels[i*l:(i+1)*l]
                    to_noise_levels = to_noise_levels[i*l:(i+1)*l]

                x_shape = self.hparams.x_shape

                xs_pred = torch.randn(
                    (batch_size, self.max_tokens, *x_shape),
                    device=self.device,
                    generator=self.generator,
                )
                xs_pred = torch.clamp(xs_pred, -self.hparams.clip_noise, self.hparams.clip_noise)

                if context is None:
                    # create empty context and zero context mask
                    context = torch.zeros_like(xs_pred)
                    context_mask = torch.zeros_like(
                        (batch_size, self.max_tokens), dtype=torch.long, device=self.device
                    )
                
                if not self.hparams.cat_context_in_c_dim:
                # replace xs_pred's context frames with context
                    xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

                if pbar is None:
                    pbar = tqdm(
                        total=scheduling_matrix.shape[0] - 1,
                        initial=0,
                        desc="Sampling with DFoT",
                        leave=False,
                    )

                # create a backup with all context tokens unmodified
                xs_pred_prev = xs_pred.clone()
                if return_all:
                    record.append(xs_pred.clone())

                conditions_mask = None

                xs_pred = self.diffusion_model.sample_step(xs_pred,
                    from_noise_levels,
                    to_noise_levels,
                    self._process_conditions(
                        conditions,
                        from_noise_levels,
                    ),
                    conditions_mask,
                    guidance_fn=None,
                )

        
        # l total length fed to the model
        # context_chunk: is the gt_frames (if we pad we get same as noise input)
        # context_mask 1 for gt_frames, 0 for noise input (same shape as input)



        # Diffusion sample this chunk:
        new_pred, _ = self._sample_sequence(
            batch_size=batch_size,
            length=l,
            context=context_chunk,
            context_mask=context_mask,
            conditions=cond_chunk,
            guidance_fn=guidance_fn,
            reconstruction_guidance=reconstruction_guidance,
            return_all=False,
            pbar=pbar,
        )




        # -----------------------------------

        # Extract only the newly generated portion from the chunk
        xs_pred = torch.cat([xs_pred, new_pred[:, -h:]], dim=1)
        curr_token = xs_pred.shape[1]
        iteration += 1

        pbar.close()
    return xs_pred[:, :length], record