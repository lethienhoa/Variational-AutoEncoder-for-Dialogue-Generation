# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Library for creating sequence-to-sequence models in TensorFlow.

Sequence-to-sequence recurrent neural networks can learn complex functions
that map input sequences to output sequences. These models yield very good
results on a number of tasks, such as speech recognition, parsing, machine
translation, or even constructing automated replies to emails.

Before using this module, it is recommended to read the TensorFlow tutorial
on sequence-to-sequence models. It explains the basic concepts of this module
and shows an end-to-end example of how to build a translation model.
  https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html

Here is an overview of functions available in this module. They all use
a very similar interface, so after reading the above tutorial and using
one of them, others should be easy to substitute.

* Full sequence-to-sequence models.
  - basic_rnn_seq2seq: The most basic RNN-RNN model.
  - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
  - embedding_rnn_seq2seq: The basic model with input embedding.
  - embedding_tied_rnn_seq2seq: The tied model with input embedding.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.

* Multi-task sequence-to-sequence models.
  - one2many_rnn_seq2seq: The embedding model with multiple decoders.

* Decoders (when you write your own encoder, you can use these to decode;
    e.g., if you want to write a model that generates captions for images).
  - rnn_decoder: The basic decoder based on a pure RNN.
  - attention_decoder: A decoder that uses the attention mechanism.

* Losses.
  - sequence_loss: Loss for a sequence model returning average log-perplexity.
  - sequence_loss_by_example: As above, but not averaging over all examples.

* model_with_buckets: A convenience function to create models with bucketing
    (see the tutorial above for an explanation of why and how to use it).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell_impl._linear  # pylint: disable=protected-access


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """

  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return loop_function


def rnn_decoder(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state


def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          num_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None):
  """RNN decoder with embedding and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
    if output_projection is not None:
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    emb_inp = (embedding_ops.embedding_lookup(embedding, i)
               for i in decoder_inputs)
    return rnn_decoder(
        emb_inp, initial_state, cell, loop_function=loop_function)

class DialogLevelLatentEncoder(self):
    """
    At the end of each utterance, the input from the utterance encoder(s) is transferred to its hidden state. 
    This hidden state is then transformed to output a mean and a (diagonal) covariance matrix,
    which parametrizes a latent Gaussian variable.
    """

    def __init__(self, input_dim, latent_dim):
	self.input_dim = input_dim
	self.latent_dim = latent_dim
	self.init_params()

    def init_params(self):
	self.Wl_deep_input = tf.Variable(tf.truncated_normal(shape=[self.input_dim, self.latent_dim], stddev=0.1), name='Wl_deep_input')
	self.bl_deep_input = tf.Variable(tf.constant(0.0, shape=[self.latent_dim]), name='bl_deep_input')

	self.Wl_in = tf.Variable(tf.truncated_normal(shape=[self.latent_dim, self.latent_dim], stddev=0.1), name='Wl_in')
	self.bl_in = tf.Variable(tf.constant(0.0, shape=[self.latent_dim]), name='bl_in')

	self.Wl_mean_out = tf.Variable(tf.truncated_normal(shape=[self.latent_dim, self.latent_dim], stddev=0.1), name='Wl_mean_out')
	self.bl_mean_out = tf.Variable(tf.constant(0.0, shape=[self.latent_dim]), name='bl_mean_out')

	self.Wl_std_out = tf.Variable(tf.truncated_normal(shape=[self.latent_dim, self.latent_dim], stddev=0.1), name='Wl_std_out')
	self.bl_std_out = tf.Variable(tf.constant(0.0, shape=[self.latent_dim]), name='bl_std_out')

	self.scale_latent_variable_variances = 0.1

    def build_encoder(self, h, x, prev_state=None):
	o_hier_info = [prev_state]		# we initialize everything to 0

	transformed_h = tf.nn.tanh( tf.add(tf.matmul(h, self.Wl_deep_input), self.bl_deep_input) )	# transform encoder
	h_out = tf.nn.tanh( tf.add(tf.matmul(transformed_h, self.Wl_in), self.bl_in) )	
	
	hs = [prev_state]
	for i in range(h_out.get_shape()[0]):		# ??
		hs = hs + h_out			# build prior on transformed encoder from the first initialization point  
	
	hs_mean = tf.add( tf.matmul(hs, self.Wl_mean_out), self.bl_mean_out )
	hs_var = tf.nn.softplus( tf.add(tf.matmul(hs, self.Wl_std_out), self.bl_std_out) ) * self.scale_latent_variable_variances

	return [hs, hs_mean, hs_var]

class DialogLevelRollLeft(self):
    """
    This class operates on hidden states at the dialogue level.
    It rolls the hidden states at utterance t to be at position t-1.
    It is used for the latent variable approximate posterior, which needs to use the future h variable.
    """

    def __init__(self, input_dim):
	self.input_dim = input_dim

    def build_encoder(self, h, x):
	hs = [prev_state]
	for i in range(h.get_shape()[0]):
		hs = hs + h
	hs = hs[::-1]

	final_hs = hs[1:(x.get_shape()[0]-1)]		# get_shape() ??
	shape = tf.get_shape(h[-1])
	final_hs = tf.concat([ final_hs, tf.reshape(h[-1], [-1,shape[0],shape[1]]) ], 0) # reshape h[-1] ??

	return final_hs


def embedding_rnn_seq2seq(encoder_inputs,
                          decoder_inputs,
                          cell,
                          num_encoder_symbols,
                          num_decoder_symbols,
                          embedding_size,
			  hiddenSize,
			  latent_gaussian_per_utterance_dim,
                          output_projection=None,
                          feed_previous=False,
                          dtype=None,
                          scope=None):
  """Embedding RNN sequence-to-sequence model.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs RNN decoder, initialized with the last
  encoder state, on embedded decoder_inputs.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial state for both the encoder and encoder
      rnn cells (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_seq2seq"

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq") as scope:
    if dtype is not None:
      scope.set_dtype(dtype)
    else:
      dtype = scope.dtype

    # Encoder.
    encoder_cell = copy.deepcopy(cell)
    encoder_cell = core_rnn_cell.EmbeddingWrapper(
        encoder_cell,
        embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    _, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype)


    # ----------------------------- Begin VAE -------------------------------------
    #
    # Initilizing Latent Variable: Prior & Posterior
    platent_utterance_variable_prior = tf.Variable(tf.constant(0.0, shape=[None, latent_gaussian_per_utterance_dim]), name="platent_utterance_variable_prior") # First dimension is num_batch
    platent_utterance_variable_approx_posterior = tf.Variable(tf.constant(0.0, shape=[None, latent_gaussian_per_utterance_dim]), name="platent_utterance_variable_approx_posterior") # First dimension is num_batch

    # Initializing prior encoder for utterance latent variable
    latent_utterance_variable_prior_encoder = DialogLevelLatentEncoder(hiddenSize, latent_gaussian_per_utterance_dim)

	# Latent variable is conditioned on the Dialogue Encoder
    print encoder_state.get_shape()		# ??
    hs_to_condition_latent_variable_on = encoder_state                  

    # Build prior encoder (conditioned on Dialogue Encoder) for utterance latent variable (transform Encoder to Latent Variable)
    _prior_out = latent_utterance_variable_prior_encoder.build_encoder(hs_to_condition_latent_variable_on, 
									encoder_inputs,    
									prev_state = platent_utterance_variable_prior)

    latent_utterance_variable_prior = _prior_out[0]
    latent_utterance_variable_prior_mean = _prior_out[1]
    latent_utterance_variable_prior_var = _prior_out[2]

    # Initializing approximate posterior encoder for utterance latent variable
    posterior_input_size = hiddenSize + hiddenSize
    latent_utterance_variable_approx_posterior_encoder = DialogLevelLatentEncoder(posterior_input_size, latent_gaussian_per_utterance_dim)	

    # Build approximate posterior encoder for utterance latent variable
    utterance_encoder_rolledleft = DialogLevelRollLeft(hiddenSize*2)	
    h_future = utterance_encoder_rolledleft.build_encoder(hs_to_condition_latent_variable_on, encoder_inputs) 
    hs_and_h_future = tf.concat([hs_to_condition_latent_variable_on, h_future], 2)

    _posterior_out = latent_utterance_variable_approx_posterior_encoder.build_encoder(hs_and_h_future, 
											encoder_inputs,				
											prev_state = platent_utterance_variable_approx_posterior)
    
    latent_utterance_variable_approx_posterior = _posterior_out[0]
    latent_utterance_variable_approx_posterior_mean = _posterior_out[1]
    latent_utterance_variable_approx_posterior_var = _posterior_out[2]

    latent_utterance_variable_approx_posterior_mean_var = tf.reduce_sum( tf.reduce_mean(latent_utterance_variable_approx_posterior_var, 2) ) / (encoder_inputs.get_shape()[0] - 1 + 0.0000001) # encoder_inputs.get_shape()[0] ??  tf.reduce_mean(latent_utterance_variable_approx_posterior_var, 2)[-1] ??

    # Compute KL divergence cost
    mean_diff_squared = (latent_utterance_variable_prior_mean - latent_utterance_variable_approx_posterior_mean) ** 2

    kl_divergences_between_prior_and_posterior  = ( 		\
						tf.reduce_sum(latent_utterance_variable_approx_posterior_var / latent_utterance_variable_prior_var, 2) 		\
						+ tf.reduce_sum( mean_diff_squared / latent_utterance_variable_prior_var, 2)		\
						- latent_gaussian_per_utterance_dim					\
						+ tf.reduce_sum( tf.log(latent_utterance_variable_prior_var), 2)	\
						- tf.reduce_sum( tf.log(latent_utterance_variable_approx_posterior_var), 2)	\
						) / 2

    kl_divergence_cost = kl_divergences_between_prior_and_posterior[:-1] # ??
    kl_divergence_cost_acc = tf.reduce_sum(kl_divergence_cost)


    # Sample utterance latent variable from posterior
    ran_cost_utterance = tf.Varialbe(tf.constant(0.0, shape=[encoder_inputs.get_shape()[0]-1]), name='ran_cost_utterance')	# shape=[encoder_inputs.get_shape()[0]-1] ??
    posterior_sample = ran_cost_utterance * tf.sqrt(latent_utterance_variable_approx_posterior_var) + latent_utterance_variable_approx_posterior_mean

    # Initialize Decoder with the combination of Encoder + Latent Variable
    encoder_state = tf.concat([encoder_state, posterior_sample], 2)	# shape = ??

    # 
    # ----------------------------- End VAE -------------------------------------

    # Decoder.
    if output_projection is None:
      cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

    if isinstance(feed_previous, bool):
      return embedding_rnn_decoder(
          decoder_inputs,
          encoder_state,
          cell,
          num_decoder_symbols,
          embedding_size,
          output_projection=output_projection,
          feed_previous=feed_previous)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse):
        outputs, state = embedding_rnn_decoder(
            decoder_inputs,
            encoder_state,
            cell,
            num_decoder_symbols,
            embedding_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(
          structure=encoder_state, flat_sequence=state_list)

    return outputs_and_state[:outputs_len], state, kl_divergence_cost_acc


def sequence_loss_by_example(kl_cost,
			     logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logit)
      else:
        crossent = softmax_loss_function(labels=target, logits=logit)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
    # Total loss = training_cost + KL_divergence_cost
    log_perps += kl_cost

  return log_perps


def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(
        sequence_loss_by_example(
            logits,
            targets,
            weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost


def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       targets,
                       weights,
                       buckets,
                       seq2seq,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(
      x, y, rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputs, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(
              sequence_loss_by_example(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))
        else:
          losses.append(
              sequence_loss(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))

  return outputs, losses

