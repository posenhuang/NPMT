-- Copyright (c) Microsoft Corporation. All rights reserved.
-- Licensed under the MIT License.
--
--[[
--
-- Dummy Criterion
--
--]]

local DummyCriterion, parent = torch.class('nn.DummyCriterion', 'nn.Criterion')

function DummyCriterion:__init()
  parent.__init(self)
end

function DummyCriterion:updateOutput(input, target)
  self.output = torch.mean(input)
  return self.output
end

function DummyCriterion:updateGradInput(input, target)
  local n = input:nElement()
  self.gradInput = input.new(input:size()):fill(1.0/n)
  return self.gradInput
end
