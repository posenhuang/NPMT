-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
-- Copyright (c) Microsoft Corporation. All rights reserved.
-- Licensed under the BSD License.
--
--[[
--
-- This model closely follows the conditional setup of rnn-lib v1, with -name
-- clstm and -aux conv_attn. See the individual functions (makeEncoder,
-- makeDecoder) for detailed comments regarding the model architecture.
--
--]]

require 'nn'
require 'nngraph'
require 'rnnlib'
local argcheck = require 'argcheck'
local mutils = require 'fairseq.models.utils'
local rutils = require 'rnnlib.mutils'
local utils = require 'fairseq.utils'

local cuda = utils.loadCuda()

local NPMTModel, parent = torch.class('NPMTModel', 'Model')

NPMTModel.make = argcheck{
    {name='self', type='NPMTModel'},
    {name='config', type='table'},
    call = function(self, config)
        config.use_cuda = true
        local encoder = self:makeEncoder(config)
        local decoder = self:makeDecoder(config)
        -- Wire up encoder and decoder
        local input = nn.Identity()()
        local sourceIn, xlength, targetIn, ylength = input:split(4)
        -- reformat the shape
        -- input to npmt is {hidden_inputs, xlength, yref, ylength}
        local output = decoder({
            encoder(sourceIn):annotate{name = 'encoder'},
            xlength,
            targetIn,
            ylength
        }):annotate{name = 'decoder'}

        return nn.gModule({input}, {output})
    end
}

-- Use the same encoder as BLSTMModel


NPMTModel.makeEncoderColumn = argcheck{
    {name='self', type='NPMTModel'},
    {name='config', type='table'},
    {name='inith', type='nngraph.Node'},
    {name='input', type='nngraph.Node'},
    {name='nlayers', type='number'},
    call = function(self, config, inith, input, nlayers)
        local rnnconfig = {
            inputsize = config.nembed,
            hidsize = config.nhid,
            nlayer = 1,
            winitfun = function(network)
                rutils.defwinitfun(network, config.init_range)
            end,
            usecudnn = usecudnn,
        }

        local rnn_class = nn.LSTM
        if config.rnn_mode == "GRU" then
          rnn_class = nn.GRU
        end

        local rnn = rnn_class(rnnconfig)
        rnn.saveHidden = false
        local output = nn.SelectTable(-1)(nn.SelectTable(2)(
            rnn({inith, input}):annotate{name = 'encoderRNN'}
        ))

        if config.use_resnet_enc then
          if config.nembed ~= config.nhid then
            local input_proj = nn.MapTable(nn.Linear(config.nembed, config.nhid, false))(input)
            output = nn.MapTable(nn.CAddTable())(nn.ZipTable()({input_proj, output}))
          else
            output = nn.MapTable(nn.CAddTable())(nn.ZipTable()({input, output}))
          end
        end

        rnnconfig.inputsize = config.nhid

        for i = 2, nlayers do
            if config.dropout_hid > 0 then
                output = nn.MapTable(nn.Dropout(config.dropout_hid))(output)
            end
            local rnn = rnn_class(rnnconfig)
            rnn.saveHidden = false
            local prev_input
            if config.use_resnet_enc then
              prev_input = nn.Identity()(output)
            end
            output = nn.SelectTable(-1)(nn.SelectTable(2)(
                rnn({
                    inith,
                    nn.ReverseTable()(output),
                })
            ))
            if config.use_resnet_enc then
              output = nn.MapTable(nn.CAddTable())(nn.ZipTable()({prev_input, output}))
            end
        end
        return output
    end
}

NPMTModel.makeEncoder = argcheck{
    doc=[[
This encoder runs a forward and backward LSTM network and concatenates their
top-most hidden states.
]],
    {name='self', type='NPMTModel'},
    {name='config', type='table'},
    call = function(self, config)
        local sourceIn = nn.Identity()()
        local inith, tokens = sourceIn:split(2)

        local dict = config.srcdict
        local lut = mutils.makeLookupTable(config, dict:size(),
            dict.pad_index)
        local embed
        if config.dropout_src > 0 then
            embed = nn.MapTable(nn.Sequential()
            :add(lut)
            :add(nn.Dropout(config.dropout_src)))(tokens)
        else
            embed = nn.MapTable(lut)(tokens)
        end
        assert(config.num_lower_conv_layers + config.num_mid_conv_layers + config.num_high_conv_layers <= 1)

        -- Low level - Add temporal conv stride to reduced computations
        if config.num_lower_conv_layers > 0 then
            local conv_embed = nn.Sequential()
            conv_embed:add(nn.MapTable(nn.View(-1, 1, config.nembed)))
            conv_embed:add(nn.JoinTable(2)) -- Split table to tensor as it expects tensor {batch_size x T x nembed}
            conv_embed:add(nn.Padding(2, 1-config.conv_kW_size))-- pad left with zeros
            conv_embed:add(nn.TemporalConvolution(config.nembed, config.nembed, config.conv_kW_size, config.conv_dW_size))
            conv_embed:add(nn.ReLU())
            embed = conv_embed(embed):annotate{name = 'TemporalConv'}
        end

        if config.num_lower_win_layers > 0 then
            local reorder_embed = nn.Sequential()
            -- Reshape as a table T elements of (batch_size x 1 x nembed)
            if config.num_lower_conv_layers == 0 then
                reorder_embed:add(nn.MapTable(nn.View(-1, 1, config.nembed)))
                reorder_embed:add(nn.JoinTable(2)) -- Split table to tensor as it expects tensor {batch_size x T x nembed}
            end

            if config.num_lower_win_layers > 0 then
              local winattn_layer
              if config.win_attn_type == 'ori' then
                  winattn_layer = nn.winAttn(config.nembed, config.kwidth, config.use_win_middle)
              else
                  winattn_layer = nil -- Error
              end
              for i = 1, config.num_lower_win_layers do
                  reorder_embed:add(winattn_layer)
              end
              embed = reorder_embed(embed)
            end
        end

        -- Mid level - Add temporal conv stride to reduced computations
        if config.num_mid_conv_layers > 0 then
            local conv_embed = nn.Sequential()
            if config.num_lower_win_layers == 0 and config.num_lower_conv_layers == 0 then
                conv_embed:add(nn.MapTable(nn.View(-1, 1, config.nembed)))
                conv_embed:add(nn.JoinTable(2)) -- Split table to tensor as it expects tensor {batch_size x T x nembed}
            end
            conv_embed:add(nn.Padding(2, 1-config.conv_kW_size))-- pad left with zeros
            conv_embed:add(nn.TemporalConvolution(config.nembed, config.nembed, config.conv_kW_size, config.conv_dW_size))
            conv_embed:add(nn.ReLU())
            embed = conv_embed(embed):annotate{name = 'TemporalConv'}
        end
        if config.num_lower_conv_layers > 0 or config.num_lower_win_layers > 0 or config.num_mid_conv_layers > 0 then
            embed = nn.SplitTable(2)(embed)
        end

        local col1 = self:makeEncoderColumn{
            config = config,
            inith = inith,
            input = embed,
            nlayers = config.nenclayer,
        }
        local col2 = self:makeEncoderColumn{
            config = config,
            inith = inith,
            input = nn.ReverseTable()(embed),
            nlayers = config.nenclayer,
        }

        -- Each column will switch direction between layers. Before merging,
        -- they should both run in the same direction (here: forward).
        if config.nenclayer % 2 == 0 then
            col1 = nn.ReverseTable()(col1)
        else
            col2 = nn.ReverseTable()(col2)
        end

        local prepare = nn.Sequential()
        -- Concatenate forward and backward states
        prepare:add(nn.JoinTable(2, 2))
        -- Scale down to nhid for further processing
        prepare:add(nn.Linear(config.nhid * 2, config.dec_unit_size, false))
        -- Add singleton dimension for subsequent joining
        prepare:add(nn.View(-1, 1, config.dec_unit_size))

        local joinedOutput = nn.JoinTable(1, 2)(
            nn.MapTable(prepare)(
                nn.ZipTable()({col1, col2})
            )
        )
        if config.dropout_hid > 0 then
            joinedOutput = nn.Dropout(config.dropout_hid)(joinedOutput)
        end

        -- TODO add attention layer     

        -- TODO add temporal conv stride to reduced computations
        if config.num_high_conv_layers > 0 then
            local conv_embed = nn.Sequential()
            conv_embed:add(nn.Padding(2, 1-config.conv_kW_size))-- pad left with zeros
            conv_embed:add(nn.TemporalConvolution(config.dec_unit_size, config.dec_unit_size, config.conv_kW_size, config.conv_dW_size))
            conv_embed:add(nn.ReLU())
            joinedOutput = conv_embed(joinedOutput):annotate{name = 'TemporalConv'}
        end

        -- avgpool_model.makeDecoder() expects two encoder outputs, one for
        -- attention score computation and the other one for applying them.
        -- We'll just use the same output for both.
        return nn.gModule({sourceIn}, {joinedOutput})
    end
}


NPMTModel.makeDecoder = argcheck{
    doc=[[
    Constructs a WASM.
    ]],
    {name='self', type='NPMTModel'},
    {name='config', type='table'},
    call = function(self, config)
       -- input to npmt is {hidden_inputs, xlength, yref, ylength}
        local input = nn.Identity()()
        local encoderOut, xlength, targetIn, ylength = input:split(4)
        local output = nn.NPMT(config)({encoderOut, xlength, targetIn, ylength}):annotate{name = 'npmt'}
        return nn.gModule({input}, {output})
    end
}


NPMTModel.prepareSource = argcheck{
    {name='self', type='NPMTModel'},
    call = function(self)
        -- Device buffers for samples
        local buffers = {
            source = {},
            xlength = {}
        }

        -- NOTE: It's assumed that all encoders start from the same hidden
        -- state.
        local encoderRNN = mutils.findAnnotatedNode(
            self:network(), 'encoderRNN'
        )
        assert(encoderRNN ~= nil)
        local conv_kW_size, conv_dW_size = 0, 0
        if mutils.findAnnotatedNode(self:network(), 'TemporalConv') then
            if #mutils.findAnnotatedNode(self:network(), 'TemporalConv') > 3 then
                conv_kW_size = mutils.findAnnotatedNode(self:network(), 'TemporalConv'):get(4).kW
                conv_dW_size = mutils.findAnnotatedNode(self:network(), 'TemporalConv'):get(4).dW
            else
                conv_kW_size = mutils.findAnnotatedNode(self:network(), 'TemporalConv'):get(2).kW
                conv_dW_size = mutils.findAnnotatedNode(self:network(), 'TemporalConv'):get(2).dW
            end
        end

        return function(sample)
            -- Encoder input
            local source = {}
            local xlength = torch.Tensor(sample.bsz):zero()
            local source_t = sample.source:t()

            local pad_index = 2
            local eos_index = 3
            local max_xlength = 0
            for i = 1, sample.bsz do
                buffers.xlength[i] = buffers.xlength[i] or torch.Tensor():type(self:type())
                xlength[i] = source_t:size(2) - torch.sum(source_t[i]:eq(pad_index))
                xlength[i] = xlength[i] - torch.sum(source_t[i]:eq(eos_index))
                max_xlength = math.max(max_xlength, xlength[i])

                source_t[{i, xlength[i]+1}] = pad_index
            end
            source_t = source_t[{{}, {1, max_xlength}}]:clone()

            for j = 1, source_t:size(2) do
                buffers.source[j] = buffers.source[j] or torch.Tensor():type(self:type())
                source[j] = mutils.sendtobuf(source_t[{{}, j}], buffers.source[j])
            end
            -- change xlength when there is a TemporalConv layer
            if conv_dW_size > 0 then
                for i = 1, sample.bsz do
                    xlength[i] = math.floor((xlength[i] - 1)/ conv_dW_size) + 1 -- Using temporal convolution
                end
            end

            local initialHidden = encoderRNN:initializeHidden(sample.bsz)
            return {{initialHidden, source}, xlength}
        end
    end
}


NPMTModel.prepareHidden = argcheck{
    {name='self', type='NPMTModel'},
    call = function(self)
        local decoderRNN = mutils.findAnnotatedNode(
            self:network(),
            'decoder'
        )
        assert(decoderRNN ~= nil)

        return function(sample)
            -- The sample contains a _cont entry if this sample is a
            -- continuation of a previous one (for truncated bptt training). In
            -- that case, start from the RNN's previous hidden state.
            if not sample._cont then
                return decoderRNN:initializeHidden(sample.bsz)
            else
                return decoderRNN:getLastHidden()
            end
        end
    end
}

NPMTModel.prepareInput = argcheck{
    {name='self', type='NPMTModel'},
    call = function(self)
        local buffers = {
            input = {},
        }

        return function(sample)
            -- Copy data to device buffers. Recurrent modules expect a table of
            -- tensors as their input.
            local input = {}
            for i = 1, sample.input:size(1) do
                buffers.input[i] = buffers.input[i]
                        or torch.Tensor():type(self:type())
                input[i] = mutils.sendtobuf(sample.input[i],
                    buffers.input[i])
            end
            return input
        end
    end
}

NPMTModel.prepareTarget = argcheck{
    {name='self', type='NPMTModel'},
    call = function(self)
        local buffers = {
            target = torch.Tensor():type(self:type()),
            ylength = torch.Tensor():type(self:type())
        }

        return function(sample)
            local target = mutils.sendtobuf(sample.target:t(), buffers.target)
            local ylength = torch.Tensor(target:size(1)):zero()

            local pad_index = 2
            local eos_index = 3
            local max_ylength = 0
            for i = 1, target:size(1) do
                ylength[i] = target:size(2) - torch.sum(target[i]:eq(pad_index))
                ylength[i] = ylength[i] - torch.sum(target[i]:eq(eos_index))
                max_ylength = math.max(ylength[i], max_ylength)
                target[{i, ylength[i]+1}] = pad_index
            end
            target = target[{{},{1,max_ylength}}]:clone()
            local ylength = mutils.sendtobuf(ylength, buffers.ylength)

            return {target, ylength}
        end
    end
}

NPMTModel.prepareSample = argcheck{
    {name='self', type='NPMTModel'},
    call = function(self)
        local prepareSource = self:prepareSource()
        local prepareTarget = self:prepareTarget()
        return function(sample)
            local source = prepareSource(sample)
            local target = prepareTarget(sample)

            local source, xlength = source[1], source[2]
            local target, ylength = target[1], target[2]
            sample.target = target
            sample.input = {source, xlength, target, ylength}
        end
    end
}


NPMTModel.generate = argcheck{
    doc=[[
Sentence generation. See search.lua for a description of search functions.
]],
    {name='self', type='Model'},
    {name='config', type='table'},
    {name='sample', type='table'},
    {name='search', type='table'},
    call = function(self, config, sample, search)
        local dict = config.dict
        local minlen = config.minlen
        local maxlen = config.maxlen
        local bsz = sample.source:size(2)
        local bbsz = config.beam * bsz
        local callbacks = self:generationCallbacks(config, bsz)
        local vocabsize = sample.targetVocab and sample.targetVocab:size(1) or dict:size()

        local timers = {
            setup = torch.Timer(),
            encoder = torch.Timer(),
            decoder = torch.Timer(),
            search_prune = torch.Timer(),
            search_results = torch.Timer(),
        }

        for k, v in pairs(timers) do
            v:stop()
            v:reset()
        end

        timers.setup:resume()
        local state = callbacks.setup(sample)
        if cuda.cutorch then
            cuda.cutorch.synchronize()
        end
        timers.setup:stop()

        timers.encoder:resume()
        callbacks.encode(state)
        timers.encoder:stop()

        timers.decoder:resume()
        local results, output_count, num_segments = callbacks.decode(state)
        if cuda.cutorch then
            cuda.cutorch.synchronize()
        end
        timers.decoder:stop()

        timers.search_results:resume()
--        local results = table.pack(search.results())
        callbacks.finalize(state, sample, results)
        timers.search_results:stop()

        local times = {}
        for k, v in pairs(timers) do
            times[k] = v:time()
        end
       -- hypos, scores, attns, t
        local attns = {}
        for i = 1, #results[2] do
            attns[i] = torch.zeros(1, vocabsize)
        end
        table.insert(results, attns)
        table.insert(results, times)
        table.insert(results, output_count)
        table.insert(results, num_segments)
        -- TODO expect hypos, scores, attns, t
        return table.unpack(results)
    end
}


NPMTModel.generationSetup = argcheck{
    {name='self', type='NPMTModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local beam = config.beam
        local bbsz = beam * bsz
        local m = self:network()
        local prepareSource = self:prepareSource()
        return function(sample)
            m:evaluate()
            local source = prepareSource(sample)
            local state = {
                sourceIn = source[1],
                xlength = source[2],
            }
            return state
        end
    end
}

NPMTModel.generationEncode = argcheck{
    {name='self', type='NPMTModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local m = self:network()
        local encoder = mutils.findAnnotatedNode(m, 'encoder')
        local beam = config.beam
        local bbsz = beam * bsz

        return function(state)
            local encoderOut = encoder:forward(state.sourceIn)

            -- There will be 'beam' hypotheses for each sentence in the batch,
            -- so duplicate the encoder output accordingly.
--            local index = torch.range(1, bsz + 1, 1 / beam)
--            index = index:narrow(1, 1, bbsz):floor():long()
--            for i = 1, encoderOut:size(1) do
--                encoderOut[i] = encoderOut[i]:index(1, index)
--            end
            state.encoderOut = encoderOut
        end
    end
}

NPMTModel.generationDecode = argcheck{
    {name='self', type='NPMTModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local m = self:network()

        local npmt = mutils.findAnnotatedNode(m, 'npmt')
        assert(npmt ~= nil)
        -- TODO add more parameters for beam search
        config.beam_size = config.beam
        config.word_weight = config.lenpen
        return function(state, targetIn)
            local output_seqs, output_probs
            local output_counts, num_segments = 0, 0
            if config.beam == 1 then
                output_seqs, output_probs, output_counts, num_segments = npmt:predict(state.encoderOut, state.xlength, config.verbose or false)
            else
                output_seqs, output_probs = npmt:beam_search(state.encoderOut, state.xlength, config)
            end
            return {output_seqs, output_probs}, output_counts, num_segments
        end
    end
}

NPMTModel.generationUpdate = argcheck{
    {name='self', type='NPMTModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local bbsz = config.beam * bsz
        local m = self:network()
        local decoderRNN = mutils.findAnnotatedNode(m, 'decoder')
        assert(decoderRNN ~= nil)

        return function(state, indexH)
            local lastH = decoderRNN:getLastHidden(bbsz)
            for i = 1, #state.prevhIn do
                for j = 1, #state.prevhIn[i] do
                    local dim = lastH[i][j]:dim() - 1
                    state.prevhIn[i][j]:copy(lastH[i][j]:index(dim, indexH))
                end
            end
        end
    end
}

function NPMTModel:float(...)
    self.module:replace(function(m)
        if torch.isTypeOf(m, 'nn.WrappedCudnnRnn') then
            return mutils.wrappedCudnnRnnToLSTMs(m)
        elseif torch.typename(m) == 'nn.SequenceTable' then
            -- Use typename() to avoid matching RecurrentTables
            return mutils.replaceCudnnRNNs(m)
        end
        return m
    end)
    return parent.float(self, ...)
end

return NPMTModel
