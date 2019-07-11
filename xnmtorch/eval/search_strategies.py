import math
from collections import namedtuple

import torch

from xnmtorch.persistence import Serializable


class SearchStrategy:

    def generate_output(self, model: 'AutoregressiveModel', initial_state: dict, src_lengths):
        raise NotImplementedError


class GreedySearch(SearchStrategy, Serializable):
    def __init__(self, max_len=100):
        self.max_len = max_len

    def generate_output(self, model: 'AutoregressiveModel', initial_state: dict, src_lengths):
        score = []
        state = initial_state
        scores = generate_fn(None, state)
        outputs = []
        for current_length in range(self.max_len):
            best_idx = torch.argmax(scores, dim=0)
            outputs.append(scores[best_idx])
            if finish_mask_fn(scores)[best_idx].item():
                break


class BeamSearch(SearchStrategy, Serializable):
    def __init__(self, beam_size, min_len=1, max_len_a=0.0, max_len_b=100, length_penalty=0.0, stop_early=True):
        self.beam_size = beam_size
        self.min_len = min_len
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.length_penalty = length_penalty
        self.stop_early = stop_early

    def generate_output(self, model: 'AutoregressiveModel', initial_state: dict, src_lengths):
        batch_size = len(src_lengths)
        max_len = max(src_lengths) * self.max_len_a + self.max_len_b
        state = initial_state

        num_remaining_samples = batch_size
        num_candidates_per_step = 2 * self.beam_size

        # batch_size, num_outputs
        all_scores = model.inference_step(None, state)

        candidate_scores, candidate_outputs = torch.topk(all_scores, num_candidates_per_step)
        candidate_beam_indices = candidate_outputs.new_zeros((batch_size, num_candidates_per_step))
        candidate_finish_mask = model.get_finish_mask(candidate_scores, candidate_outputs)

        model.reorder_state(state, torch.arange(batch_size, dtype=candidate_outputs.dtype,
                                                device=candidate_outputs.device)
                            .repeat_interleave(self.beam_size))

        # Cumulative best scores of unfinished beams
        scores = candidate_scores.new_full((batch_size * self.beam_size,), 0)
        # History of best outputs of unfinished beams
        outputs = candidate_outputs.new_full((batch_size * self.beam_size, max_len), 0)
        # Back buffer, will be swapped with outputs
        outputs_buf = outputs.clone()

        batch_beam_offsets = (torch.arange(batch_size, dtype=outputs.dtype, device=outputs.device)
                              * self.beam_size).unsqueeze(1)
        candidate_offsets = torch.arange(num_candidates_per_step, dtype=outputs.dtype, device=outputs.device)

        search_outputs = [[] for _ in range(batch_size)]
        is_sample_finished = [False for _ in range(batch_size)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for _ in range(batch_size)]

        # helper function for allocating buffers on the fly
        def buffer(type_of=outputs):
            return type_of.new()

        finished_batch_beam_indices = buffer()
        finished_scores = buffer(type_of=scores)
        finished_outputs = buffer()
        active_mask = buffer()  # will actually hold ints, so should be type long, not uint8
        active_indices = buffer()
        _ignore = buffer()
        active_batch_beam_indices = buffer()

        def check_sample_finished(sample_idx, step, raw_sample_idx, candidate_scores=None):
            if len(search_outputs[sample_idx]) == self.beam_size:
                if self.stop_early or step == max_len - 1 or candidate_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = candidate_scores[raw_sample_idx].max()
                if self.length_penalty != 0.0:
                    best_unfinalized_score /= max_len ** self.length_penalty
                if worst_finalized[sample_idx]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_candidates(step, finished_batch_beam_indices, finished_scores, finished_outputs, candidate_scores=None):
            assert finished_batch_beam_indices.numel() == finished_scores.numel()

            finished_strings = outputs.index_select(0, finished_batch_beam_indices)[:, :step+1]
            finished_strings[:, step] = finished_outputs

            # some samples in the batch may be completely finished
            unfinished_offset = []
            prev = 0
            for f in is_sample_finished:
                if f:
                    prev += 1
                else:
                    unfinished_offset.append(prev)

            samples_seen = set()
            for i, (index, score) in enumerate(zip(finished_batch_beam_indices.tolist(), finished_scores.tolist())):
                raw_sample_index = index // self.beam_size
                sample_idx = raw_sample_index + unfinished_offset[raw_sample_index]

                samples_seen.add((sample_idx, raw_sample_index))

                if self.length_penalty != 0.0:
                    finished_scores /= step ** self.length_penalty

                if len(search_outputs[sample_idx]) < self.beam_size:
                    search_outputs[sample_idx].append({"outputs": finished_strings[i], "score": score})
                elif not self.stop_early and score > worst_finalized[sample_idx]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sample_idx]['idx']
                    if worst_idx is not None:
                        search_outputs[sample_idx][worst_idx] = {"outputs": finished_strings[i], "score": score}

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(search_outputs[sample_idx]), key=lambda r: r[1]['score'])
                    worst_finalized[sample_idx] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            newly_finished = []
            for sample_idx, raw_sample_index in samples_seen:
                if not is_sample_finished[sample_idx] and \
                        check_sample_finished(sample_idx, step, raw_sample_index, candidate_scores):
                    is_sample_finished[sample_idx] = True
                    newly_finished.append(raw_sample_index)
            return newly_finished

        for step in range(max_len):
            candidate_batch_beam_indices = candidate_beam_indices + batch_beam_offsets

            finalized_samples = []
            if step >= self.min_len:
                reduced_finished_mask = candidate_finish_mask[:, :self.beam_size]
                if reduced_finished_mask.sum().item() > 0:
                    torch.masked_select(
                        candidate_batch_beam_indices[:, :self.beam_size],
                        reduced_finished_mask,
                        out=finished_batch_beam_indices
                    )
                    torch.masked_select(
                        candidate_scores[:, :self.beam_size],
                        reduced_finished_mask,
                        out=finished_scores
                    )
                    torch.masked_select(
                        candidate_outputs[:, :self.beam_size],
                        reduced_finished_mask,
                        out=finished_outputs
                    )
                    finalized_samples = finalize_candidates(step, finished_batch_beam_indices,
                                                            finished_scores, finished_outputs, candidate_scores)
                    num_remaining_samples -= len(finalized_samples)

            assert num_remaining_samples >= 0
            if num_remaining_samples == 0:
                break

            if len(finalized_samples) > 0:
                batch_mask = candidate_outputs.new_ones((batch_size,))
                batch_mask[candidate_outputs.new_tensor(finalized_samples)] = 0
                batch_indices = batch_mask.nonzero().squeeze(-1)

                candidate_finish_mask = candidate_finish_mask[batch_indices]
                candidate_beam_indices = candidate_beam_indices[batch_indices]
                batch_beam_offsets.resize_(num_remaining_samples, 1)
                candidate_batch_beam_indices = candidate_beam_indices + batch_beam_offsets
                candidate_scores = candidate_scores[batch_indices]
                candidate_outputs = candidate_outputs[batch_indices]
                src_lengths = src_lengths[batch_indices]

                outputs = outputs.view(batch_size, -1)[batch_indices].view(num_remaining_samples * self.beam_size, -1)
                outputs_buf.resize_as_(outputs)

                scores = scores.view(batch_size, -1)[batch_indices].view(-1)

                assert num_remaining_samples == batch_size - len(finalized_samples)
                batch_size = num_remaining_samples
            else:
                batch_indices = None

            # candidate_offsets is just [0,1,2,3...,num_candidates_per_step-1]
            # by adding num_candidates_per_step to the finished indices, then selecting the lowest values,
            # we select the hypotheses that are at the front of the array, but not finished
            torch.add(
                candidate_finish_mask.type_as(candidate_offsets) * num_candidates_per_step,
                candidate_offsets[:candidate_finish_mask.size(1)],
                out=active_mask
            )

            # The hypotheses are sorted by probability, so we want to pick the first beam_size ones
            # that are not finished
            torch.topk(
                active_mask, k=self.beam_size, dim=1, largest=False,
                out=(_ignore, active_indices)
            )

            torch.gather(
                candidate_batch_beam_indices, dim=1, index=active_indices,
                out=active_batch_beam_indices
            )

            active_batch_beam_indices = active_batch_beam_indices.view(-1)

            torch.index_select(
                outputs[:, :step], dim=0, index=active_batch_beam_indices,
                out=outputs_buf[:, :step]
            )
            torch.gather(
                candidate_outputs, dim=1, index=active_indices,
                out=outputs_buf.view(batch_size, self.beam_size, -1)[:, :, step]  # +1?
            )
            torch.gather(
                candidate_scores, dim=1, index=active_indices,
                out=scores.view(batch_size, self.beam_size)
            )
            outputs, outputs_buf = outputs_buf, outputs

            if step < max_len - 1:
                reorder_state = active_batch_beam_indices
                if batch_indices is not None:
                    # update beam indices to take into account removed samples
                    correction = batch_indices - torch.arange(batch_indices.numel(), dtype=batch_indices.dtype,
                                                              device=batch_indices.device)
                    reorder_state.view(-1, self.beam_size).add_(correction.unsqueeze(-1) * self.beam_size)

                # reorder decoder internal states based on the prev choice of beams
                model.reorder_state(state, reorder_state)

                # batch_size*beam_size, vocab_size
                all_scores = model.inference_step(outputs[:, step], state)

                # make probs contain cumulative scores for each hypothesis
                all_scores.add_(scores.unsqueeze(-1))

                num_classes = all_scores.size(-1)

                torch.topk(
                    all_scores.view(batch_size, -1),
                    k=num_candidates_per_step,
                    out=(candidate_scores, candidate_outputs)
                )
                candidate_beam_indices = torch.div(candidate_outputs, num_classes)
                candidate_outputs.fmod_(num_classes)

                candidate_finish_mask = model.get_finish_mask(candidate_scores, candidate_outputs)

        if num_remaining_samples > 0:
            finish_indices = []
            i = 0
            for outs in search_outputs:
                required = self.beam_size - len(outs)
                if required > 0:
                    for j in range(required):
                        finish_indices.append(batch_beam_offsets[i] + j)
                    i += 1
            finish_indices = outputs.new_tensor(finish_indices)
            finish_scores = scores.index_select(0, finish_indices)
            finish_outputs = outputs[:, -1].index_select(0, finish_indices)
            num_remaining_samples -= len(finalize_candidates(max_len - 1, finish_indices, finish_scores,
                                                             finish_outputs))
            assert num_remaining_samples == 0

        for sample in range(len(search_outputs)):
            search_outputs[sample] = sorted(search_outputs[sample], key=lambda r: r["score"], reverse=True)

        return search_outputs
