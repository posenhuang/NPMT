-- Copyright (c) Microsoft Corporation. All rights reserved.
-- Licensed under the MIT License.
--
--[[
--
-- Input: a tensor of input: eq_length * batch_size * input_dim or a table of h0 and input
-- where h0: num_layer * batch_size * hidden_dim; the first dimension optional if num_layer == 1
-- Output: a tensor of eq_length * batch_size * output_dim
--
--]]
--

require("nn")
require("cudnn")
require("cutorch")

local mRNN, parent = torch.class('nn.mRNN', 'nn.Container')

function mRNN:__init(input_dim, hidden_dim, bd, mode, has_otherOutput, use_resnet, use_skip_mode, dropout, add_output)
  parent.__init(self)
  local batchFirst = batchFirst or true
  assert(batchFirst == true)

  if use_resnet then
    print("resnet is used in current layer")
  end

  self.bd = bd or false
  self.mode = mode or "CUDNN_GRU"
  self.has_otherOutput = has_otherOutput or false
  self.dropout = dropout or 0
  self.add_output = add_output or false
  self.use_resnet = use_resnet or false

  local rnn_hidden_dim = hidden_dim
  if self.bd then
    if self.add_output then
      print("bd rnn output is added")
    else
      rnn_hidden_dim = rnn_hidden_dim / 2
    end
  end

  self.rnn_hidden_dim = rnn_hidden_dim
  local rnn = cudnn.RNN(input_dim, rnn_hidden_dim, 1, batchFirst)
  if use_skip_mode then
    assert(not self.bd and input_dim == hidden_dim)
    rnn.inputMode = 'CUDNN_SKIP_INPUT'
  end
  self.rnn = rnn

  rnn.mode = self.mode
  if self.bd then
    rnn.numDirections = 2
    rnn.bidirectional = 'CUDNN_BIDIRECTIONAL'
  end
  rnn:reset()
  self:add(rnn)
  if use_resnet and input_dim ~= hidden_dim then
    self.input_proj = nn.Bottle(nn.Linear(input_dim, hidden_dim, false))
    self:add(self.input_proj)
  end
end

function mRNN:setStates(h0, c0)
  self.rnn.hiddenInput = h0:clone()
  if c0 then
    self.rnn.cellInput = c0:clone()
  end
end

function mRNN:getNextMemInput() 
  if self.rnn.cellOutput then
    return self.rnn.cellOutput:clone()
  else
    return self.rnn.hiddenOutput:clone()
  end
end

function mRNN:updateOutput(input)
  self.recompute_backward = true
  local c0, h0, x
  if torch.type(input) == "table" then
    assert(not self.bd)
    if #input == 2 then
      h0, x = unpack(input)
      if (h0:dim() == 2) then
        h0 = h0:view(1, h0:size(1), h0:size(2))
      end
      if self.mode == "CUDNN_LSTM" then
        self.rnn.cellInput = h0
      else
        self.rnn.hiddenInput = h0
      end
    elseif #input == 3 then
      c0, h0, x = unpack(input)
      if (h0:dim() == 2) then
        h0 = h0:view(1, h0:size(1), h0:size(2))
      end
      if (c0:dim() == 2) then
        c0 = c0:view(1, c0:size(1), c0:size(2))
      end
      self.rnn.hiddenInput = h0
      self.rnn.cellInput = c0
    end
  else
    x = input
  end
  local rnn_output = self.rnn:updateOutput(x)
  if self.bd and self.add_output then
    rnn_output = torch.add(rnn_output[{{}, {}, {1,self.rnn_hidden_dim}}], 
                           rnn_output[{{}, {}, {self.rnn_hidden_dim+1,2*self.rnn_hidden_dim}}])
  end
  local output = rnn_output
  if self.use_resnet then
    if self.input_proj then
      output = torch.add(output, self.input_proj:updateOutput(x))
    else
      output = torch.add(output, x)
    end
  end

  if self.has_otherOutput then
    local otherOutput
    if self.mode == "CUDNN_LSTM" then
      otherOutput = self.rnn.cellOutput:clone()
    else
      otherOutput = self.rnn.hiddenOutput:clone()
    end
    assert(otherOutput:dim() == 3)
    if self.bd then
      if self.add_output then
        otherOutput = torch.add(otherOutput[1], otherOutput[2])
      else
        otherOutput = torch.cat(otherOutput[1], otherOutput[2], 2)
      end
    else
      assert(otherOutput:size(1) == 1)
      otherOutput = otherOutput[1] 
    end
    assert(otherOutput:dim() == 2)
    self.output = {output, otherOutput}
  else
    self.output = output
  end
  return self.output
end

function mRNN:backward(input, gradOutput, scale)
  scale = scale or 1
  self.recompute_backward = false
  if self.has_otherOutput then
    local otherGradOutput
    gradOutput, otherGradOutput = unpack(gradOutput)
    assert(otherGradOutput:dim() == 2)
    if self.bd then
      local forwardGradOutput, backwardGradOutput
      if self.add_output then
        forwardGradOutput = otherGradOutput
        backwardGradOutput = otherGradOutput
      else
        forwardGradOutput, backwardGradOutput = unpack(torch.chunk(otherGradOutput, 2, 2))
      end
      otherGradOutput = torch.cat(forwardGradOutput:reshape(1, forwardGradOutput:size(1), forwardGradOutput:size(2)),
                                  backwardGradOutput:reshape(1, backwardGradOutput:size(1), backwardGradOutput:size(2)), 1)
    else
      otherGradOutput = otherGradOutput:view(1, otherGradOutput:size(1), otherGradOutput:size(2))
    end
    assert(otherGradOutput:dim() == 3)
    if self.mode == "CUDNN_LSTM" then
      self.rnn.gradCellOutput = otherGradOutput
    else
      self.rnn.gradHiddenOutput = otherGradOutput
    end
  end

  local h0, c0, x
  if torch.type(input) == "table" then
    if #input == 2 then
      h0, x = unpack(input)
    elseif #input == 3 then
      c0, h0, x = unpack(input)
    end
  else
    x = input
  end

  local gradInput = x.new(x:size()):zero()
  if self.use_resnet then
    if self.input_proj then
      gradInput:add(self.input_proj:backward(x, gradOutput))
    else
      gradInput:add(gradOutput)
    end
  end

  if self.bd and self.add_output then
    gradInput:add(self.rnn:backward(x, torch.cat(gradOutput, gradOutput, 3), scale))
  else
    gradInput:add(self.rnn:backward(x, gradOutput, scale))
  end

  local h0_grad, c0_grad
  if h0 and c0 then
    h0_grad = self.rnn.gradHiddenInput
    c0_grad = self.rnn.gradCellInput
    self.gradInput = {c0_grad:view(h0:size()), h0_grad:view(c0:size()), gradInput}
  elseif h0 then
    if self.mode == "CUDNN_LSTM" then
      h0_grad = self.rnn.gradCellInput
    else
      h0_grad = self.rnn.gradHiddenInput
    end
    self.gradInput = {h0_grad:view(h0:size()), gradInput}
  else 
    self.gradInput = gradInput
  end
  return self.gradInput
end

function mRNN:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput, 1.0)
  end
  return self.gradInput
end

function mRNN:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end

function mRNN:clearState()
  parent.clearState(self)
  self.rnn:resetStates()
end

function mRNN:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.Container'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end
