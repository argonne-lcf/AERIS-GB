# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.

# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions
# This work is licensed under the MIT License. See LICENSE for details.

from deepspeed.runtime.pipe.schedule import LoadMicroBatch, SendActivation, RecvActivation, ForwardPass, BufferOpInstruction, PipeSchedule, SendGrad, RecvGrad, BackwardPass, ReduceTiedGrads, ReduceGrads, OptimizerStep, _is_even, _is_odd

class CustomTrainSchedule(PipeSchedule):
    """Inspired by DeepSpeed schedules

    """
    def steps(self):
        """"""
        total_steps = 2 * (self.micro_batches + self.stages - 1) + 1
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            prev_micro_batch_id, prev_forward = self._step_to_micro_batch(step_id-1)
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)
            next_micro_batch_id, next_forward = self._step_to_micro_batch(step_id+1)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)
            if self._valid_micro_batch(next_micro_batch_id):
                next_buffer = self._buffer_idx(next_micro_batch_id)

            cmds = []


            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))
            
            
            #if (not is_forward) and 
            
            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(ForwardPass(curr_buffer))
                    #Avoid deadlock this way because D2H async was hard on XPU. Usually this would be before RecvActivation 
                    if self._valid_stage(self.next_stage):
                        cmds.append(SendActivation(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))
            
            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append(SendGrad(prev_buffer))
                if self._valid_micro_batch(next_micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append(RecvGrad(next_buffer))
            else:
                #if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.next_stage):
                #cmds.append(SendActivation(prev_buffer))
                if self._valid_micro_batch(next_micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append(RecvActivation(next_buffer))
            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

    def num_pipe_buffers(self):
        """Return the number of pipeline buffers required for this stage.

        This is equivalent to the maximum number of in-flight forward passes,
        since we need to remember the activations of forward passes in order
        to run backpropagation. For synchronous 1F1B, this is equivalent to
        the index difference between this stage and the last stage.
        """
        buffers = min(self.stages - self.stage_id, self.micro_batches)+1
        return max(2, buffers)

    def _step_to_micro_batch(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id


class SubmitRecvGrad(BufferOpInstruction):
    """Compute a forward pass.

    Roughly:

    .. code-block:: python

        buffers['outputs'][buffer_id] = forward(buffers['inputs'][buffer_id])
    """
    pass

class SubmitRecvActivation(BufferOpInstruction):
    """Compute a forward pass.

    Roughly:

    .. code-block:: python

        buffers['outputs'][buffer_id] = forward(buffers['inputs'][buffer_id])
    """
    pass

class ProcessActivation(BufferOpInstruction):
    """Compute a forward pass.

    Roughly:

    .. code-block:: python

        buffers['outputs'][buffer_id] = forward(buffers['inputs'][buffer_id])
    """
    pass

class ProcessGrad(BufferOpInstruction):
    """Compute a forward pass.

    Roughly:

    .. code-block:: python

        buffers['outputs'][buffer_id] = forward(buffers['inputs'][buffer_id])
    """
    pass

class LessCustomTrainSchedule(PipeSchedule):
    
    """Inspired by DeepSpeed schedules

    """

    def steps(self):
        """"""
        total_steps = 2 * (self.micro_batches + self.stages - 1) + 1
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            prev_micro_batch_id, prev_forward = self._step_to_micro_batch(step_id-1)
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)
            next_micro_batch_id, next_forward = self._step_to_micro_batch(step_id+1)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)
            if self._valid_micro_batch(next_micro_batch_id):
                next_buffer = self._buffer_idx(next_micro_batch_id)

            cmds = []

            
            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            if is_forward:
                if self._valid_micro_batch(next_micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append(SubmitRecvGrad(next_buffer))
            else:
                if self._valid_micro_batch(next_micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append(SubmitRecvActivation(next_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    if self._valid_stage(self.prev_stage):
                        cmds.append(ProcessActivation(curr_buffer))
                    cmds.append(ForwardPass(curr_buffer))
                    #Avoid deadlock this way because D2H async was hard on XPU. Usually this would be before RecvActivation 
                    if self._valid_stage(self.next_stage):
                        cmds.append(SendActivation(curr_buffer))
                else:
                    if self._valid_stage(self.next_stage):
                        cmds.append(ProcessGrad(curr_buffer))
                    cmds.append(BackwardPass(curr_buffer))
                    if self._valid_stage(self.prev_stage):
                        cmds.append(SendGrad(curr_buffer))
            
            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

    def num_pipe_buffers(self):
        """Return the number of pipeline buffers required for this stage.

        This is equivalent to the maximum number of in-flight forward passes,
        since we need to remember the activations of forward passes in order
        to run backpropagation. For synchronous 1F1B, this is equivalent to
        the index difference between this stage and the last stage.
        """
        buffers = min(self.stages - self.stage_id, self.micro_batches)+1
        return max(2, buffers)

    def _step_to_micro_batch(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id
