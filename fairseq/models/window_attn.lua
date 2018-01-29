-- Copyright (c) Microsoft Corporation. All rights reserved.
-- Licensed under the MIT License.
--
--[[
--
-- Reordering layer
--
--]]
--
require("nn")
require("nngraph")

local winAttn, parent = torch.class('nn.winAttn', 'nn.Container')

function make_win_unit(input_size, kW)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  local x = unpack(inputs) -- B * kW * d
  local reshaped_x = nn.Reshape(input_size*kW, true)(x) -- B * (kW*d)
  local weight = nn.Sigmoid()(nn.Linear(input_size*kW, kW)(reshaped_x)):annotate{name = 'winAtt_weight'} -- B * kW
  weight = nn.Replicate(input_size, 3)(weight) -- B * kW * d
  local output = nn.CMulTable()({x, weight}) -- B * kW * d
  output = nn.Tanh()(nn.Sum(2)(output)) -- B * d
  return nn.gModule(inputs, {output})
end

function winAttn:__init(input_size, kW, use_middle)
  parent.__init(self)
  self.gradInput = torch.Tensor()
  self.output = torch.Tensor()
  self.padded_input = torch.Tensor()

  self.input_size = input_size
  self.kW = kW
  if use_middle then
    local width = math.floor(self.kW / 2)
    self.kW = width * 2 + 1
    self.padding = nn.Sequential():add(nn.Padding(2, -width)):add(nn.Padding(2, width))
  else
    self.padding = nn.Padding(2, 1 - self.kW)
  end

  self:add(self.padding)
  self.win_unit = make_win_unit(input_size, self.kW)
  self:add(self.win_unit)
  self.win_unit_clones = {}
  self.max_T = 0
end

function winAttn:updateOutput(input)
  self.recompute_backward = true

  local T = input:size(2)
  if self.max_T < T then
    self.win_unit:clearState()
    local more_win_units = g_cloneManyTimes(self.win_unit, T - self.max_T)
    for i = 1, T - self.max_T do
      table.insert(self.win_unit_clones, more_win_units[i])
    end
    self.max_T = T
  end
  for t = 1, T do
    self.win_unit_clones[t]:clearState()
  end
  self.padded_input = self.padding:updateOutput(input)
  self.output = input.new(input:size()):zero()
--  local mutils = require 'fairseq.models.utils'
  for t = 1, T do
    local x = self.padded_input[{{}, {t, t+self.kW-1}, {}}]
    local y = self.win_unit_clones[t]:updateOutput(x)
    self.output[{{}, t, {}}]:add(y)
--    local weight =  mutils.findAnnotatedNode(self.win_unit_clones[t], 'winAtt_weight')
--    print('t', t, 'weight', weight.output)
--    print(self.win_unit_clones[t].forwardnodes[6].data.input[1])
  end
  return self.output
end

function winAttn:backward(input, gradOutput, scale)
  local scale = scale or 1
  self.recompute_backward = false
  local grad_padded_input = self.padded_input.new(self.padded_input:size()):zero()
  local T = input:size(2)
  for t = 1, T do
    local x = self.padded_input[{{}, {t, t+self.kW-1}, {}}]
    local grad_x = self.win_unit_clones[t]:backward(x, gradOutput[{{}, t, {}}])
    grad_padded_input[{{}, {t, t+self.kW-1}, {}}]:add(grad_x)
  end
  self.gradInput = self.padding:backward(input, grad_padded_input, scale)
  return self.gradInput
end

function winAttn:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput, 1.0)
  end
  return self.gradInput
end

function winAttn:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end

function winAttn:training()
  parent.training(self)
  for t = 1, self.max_T do
    self.win_unit_clones[t]:training()
  end
end

function winAttn:evaluate()
  parent.evaluate(self)
  for t = 1, self.max_T do
    self.win_unit_clones[t]:evaluate()
  end
end

function winAttn:clearState()
  parent.clearState(self)
  self.output:set()
  self.padded_input:set()
  self.gradInput:set()
  for t = 1, self.max_T do
    self.win_unit_clones[t]:clearState()
  end
end
