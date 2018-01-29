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
-- Main training script.
--
--]]

require 'rnnlib'
require 'xlua'
require 'optim'
require 'fairseq'

local tnt = require 'torchnet'
local plpath = require 'pl.path'
local pltablex = require 'pl.tablex'
local hooks = require 'fairseq.torchnet.hooks'
local data = require 'fairseq.torchnet.data'
local search = require 'fairseq.search'
local utils = require 'fairseq.utils'

local cuda = utils.loadCuda()
-- we require cuda for training
assert(cuda.cutorch)

local cmd = torch.CmdLine()
cmd:option('-sourcelang', 'de', 'source language')
cmd:option('-targetlang', 'en', 'target language')
cmd:option('-datadir', 'data-bin')
cmd:option('-model', 'avgpool', 'model type {avgpool|blstm|bgru|conv|fconv|npmt}')
cmd:option('-nembed', 256, 'dimension of embeddings and attention')
cmd:option('-noutembed', 256, 'dimension of the output embeddings')
cmd:option('-nhid', 256, 'number of hidden units per layer')
cmd:option('-nlayer', 1, 'number of hidden layers in decoder')
cmd:option('-nenclayer', 1, 'number of hidden layers in encoder')
cmd:option('-nagglayer', -1,
    'number of layers for conv encoder aggregation stack (CNN-c)')
cmd:option('-kwidth', 3, 'kernel width for conv encoder')
cmd:option('-klmwidth', 3, 'kernel width for convolutional language models')
cmd:option('-optim', 'sgd', 'optimization algortihm {sgd|adam|nag}')
-- See note about normalization and hyper-parameters below
cmd:option('-timeavg', false,
    'average gradients over time (as well as sequences)')
cmd:option('-lr', 0.1, 'learning rate (per time step without -timeavg)')
cmd:option('-lrshrink', 10, 'learning rate shrinking factor for annealing')
cmd:option('-momentum', 0, 'momentum for sgd/nag optimizers')
cmd:option('-annealing_type', 'fast',
    'whether to decrease learning rate with a fast or slow schedule')
cmd:option('-noearlystop', false, 'no early stopping for Adam/Adagrad')
cmd:option('-batchsize', 32, 'batch size (number of sequences)')
cmd:option('-bptt', 25, 'back-prop through time steps')
cmd:option('-maxbatch', 0, 'maximum number of tokens per batch')
cmd:option('-clip', 25,
    'clip threshold of gradients (per sequence without -timeavg)')
cmd:option('-maxepoch', 100, 'maximum number of epochs')
cmd:option('-minepochtoanneal', 0, 'minimum number of epochs before annealing')
cmd:option('-maxsourcelen', 0,
    'maximum source sentence length in training data')
cmd:option('-ndatathreads', 1, 'number of threads for data preparation')
cmd:option('-log_interval', 1000, 'log training statistics every n updates')
cmd:option('-save_interval', -1,
    'save snapshot every n updates (defaults to once per epoch)')
cmd:option('-init_range', 0.05, 'range for random weight initialization')
cmd:option('-savedir', '.', 'save models here')
cmd:option('-nosave', false, 'don\'t save models and checkpoints')
cmd:option('-nobleu', false, 'don\'t produce final BLEU scores')
cmd:option('-notext', false, 'don\'t produce final generation output')
cmd:option('-validbleu', false, 'produce validation BLEU scores on checkpoints')
cmd:option('-log', false, 'whether to enable structured logging')
cmd:option('-seed', 1111, 'random number seed')
cmd:option('-aligndictpath', '', 'path to an alignment dictionary (optional)')
cmd:option('-nmostcommon', 500,
    'the number of most common words to keep when using alignment')
cmd:option('-topnalign', 100, 'the number of the most common alignments to use')
cmd:option('-freqthreshold', 0,
    'the minimum frequency for an alignment candidate in order' ..
    'to be considered (default no limit)')
cmd:option('-ngpus', cuda.cutorch:getDeviceCount(),
    'number of gpus for data parallel training')
cmd:option('-dropout_src', -1, 'dropout on source embeddings')
cmd:option('-dropout_tgt', -1, 'dropout on target embeddings')
cmd:option('-dropout_out', -1, 'dropout on decoder output')
cmd:option('-dropout_hid', -1, 'dropout between layers')
cmd:option('-dropout', 0, 'set negative dropout_* options to this value')

-- Options for fconv_model
cmd:option('-cudnnconv', false, 'use cudnn.TemporalConvolution (slower)')
cmd:option('-attnlayers', '-1', 'decoder layers with attention (-1: all)')
cmd:option('-bfactor', 0, 'factor to divide nhid in bottleneck structure')
cmd:option('-fconv_nhids', '',
    'comma-separated list of hidden units for each encoder layer')
cmd:option('-fconv_nlmhids', '',
    'comma-separated list of hidden units for each decoder layer')
cmd:option('-fconv_kwidths', '',
    'comma-separated list of kernel widths for conv encoder')
cmd:option('-fconv_klmwidths', '',
    'comma-separated list of kernel widths for convolutional language model')

-- Options for NPMT
cmd:option('-max_segment_len', 6, 'maximum segment length in the output')
cmd:option('-num_lower_win_layers', 0, 'reorder layer')
cmd:option('-use_win_middle', true, 'reorder layer with window centered at t')
cmd:option('-dec_unit_size', 256, 'number of hidden units per layer in decoder (uni-directional)')
cmd:option('-word_weight', 0.5, 'Use word weight.')
cmd:option('-lm_weight', 0.0, 'external lm weight.')
cmd:option('-lm_path', "", 'external lm path.')
cmd:option('-use_resnet_enc', false, 'use resnet connections in enc')
cmd:option('-use_resnet_dec', false, 'use resnet connections in dec')
cmd:option('-npmt_dropout', 0, 'npmt dropout factor')
cmd:option('-rnn_mode', "LSTM", 'or GRU')
cmd:option('-use_cuda', true, 'use cuda')
cmd:option('-beam', 10, 'beam size')
cmd:option('-group_size', 512, 'group size')
cmd:option('-use_accel', false, 'use C++/CUDA acceleration')
cmd:option('-conv_kW_size', 3, 'kernel width for temporal conv layer')
cmd:option('-conv_dW_size', 2, 'kernel stride for temporal conv layer')
cmd:option('-num_lower_conv_layers', 0, 'num lower temporal conv layers')
cmd:option('-num_mid_conv_layers', 0, 'num mid temporal conv layers')
cmd:option('-num_high_conv_layers', 0, 'num higher temporal conv layers')
cmd:option('-win_attn_type', 'ori', 'ori: original')
cmd:option('-reset_lrate', false, 'True reset learning rate after reloading')
cmd:option('-use_nnlm', false, 'True use a separated RNN')



--chowang: we don't need the following anymore?
--cmd:option('-unk_symbol', 1, 'unk symbol id')
--cmd:option('-start_symbol', 2, 'start symbol id')
--cmd:option('-end_symbol', 3, 'end symbol id')

local config = cmd:parse(arg)

if config.dropout_src < 0 then config.dropout_src = config.dropout end
if config.dropout_tgt < 0 then config.dropout_tgt = config.dropout end
if config.dropout_out < 0 then config.dropout_out = config.dropout end
if config.dropout_hid < 0 then config.dropout_hid = config.dropout end

-- parse hidden sizes and kernel widths
if config.model == 'fconv' then
    -- encoder
    config.nhids = utils.parseListOrDefault(
        config.fconv_nhids, config.nenclayer, config.nhid)
    config.kwidths = utils.parseListOrDefault(
        config.fconv_kwidths, config.nenclayer, config.kwidth)

    -- deconder
    config.nlmhids = utils.parseListOrDefault(
        config.fconv_nlmhids, config.nlayer, config.nhid)
    config.klmwidths = utils.parseListOrDefault(
        config.fconv_klmwidths, config.nlayer, config.klmwidth)
end

torch.manualSeed(config.seed)
cuda.cutorch.manualSeed(config.seed)


assert(config.ngpus >= 1 and config.ngpus <= cuda.cutorch.getDeviceCount())

-- Effective batchsize equals to the base batchsize * ngpus
config.batchsize = config.batchsize * config.ngpus
config.maxbatch = config.maxbatch * config.ngpus
print(config)

-------------------------------------------------------------------
-- Load data
-------------------------------------------------------------------
config.dict = torch.load(plpath.join(config.datadir,
    'dict.' .. config.targetlang .. '.th7'))
print(string.format('| [%s] Dictionary: %d types', config.targetlang,
    config.dict:size()))
config.srcdict = torch.load(plpath.join(config.datadir,
    'dict.' .. config.sourcelang .. '.th7'))
print(string.format('| [%s] Dictionary: %d types', config.sourcelang,
    config.srcdict:size()))

if config.aligndictpath ~= '' then
    config.aligndict = tnt.IndexedDatasetReader{
        indexfilename = config.aligndictpath .. '.idx',
        datafilename = config.aligndictpath .. '.bin',
        mmap = true,
        mmapidx = true,
    }
    config.nmostcommon = math.max(config.nmostcommon, config.dict.nspecial)
    config.nmostcommon = math.min(config.nmostcommon, config.dict:size())
end

local train, test = data.loadCorpus{
    config = config,
    trainsets = {'train'},
    testsets = {'valid'},
}
local corpus = {
    train = train.train,
    valid = test.valid,
    test = test.test,
}

-------------------------------------------------------------------
-- Setup models and training criterions
-------------------------------------------------------------------
local seed = config.seed
local thread_init_fn = function(id)
    require 'nn'
    require 'cunn'
    require 'nngraph'
    require 'rnnlib'
    require 'fairseq.models.utils'
    require 'fairseq'
    require 'fairseq.torchnet'
    require 'threads'
    require 'torchnet'
    require 'argcheck'
    require 'cutorch'
    -- Make sure we have a different seed for each thread so that random
    -- pertubations during training (e.g. dropout) are different for each
    -- worker.
    torch.manualSeed(seed + id - 1)
    cutorch.manualSeed(seed + id - 1)
end

local make_model_fn = function(id)
    local model = require(
        string.format('fairseq.models.%s_model',
            config.model)
    ).new(config)
    model:cuda()
    return model
end

local make_criterion_fn = function(id)
    -- Don't produce losses and gradients for the padding symbol
    local padindex = config.dict:getIndex(config.dict.pad)
    local critweights = torch.ones(config.dict:size()):cuda()
    critweights[padindex] = 0
    local criterion
    if config.model ~= 'npmt' then
        criterion = nn.CrossEntropyCriterion(critweights, false):cuda()
    else
        criterion = nn.DummyCriterion(critweights, false):cuda()
    end
    return criterion, critweights
end

-------------------------------------------------------------------
-- Torchnet engine setup
-------------------------------------------------------------------
engine = tnt.ResumableDPOptimEngine(
    config.ngpus, thread_init_fn, make_model_fn, make_criterion_fn
)
local lossMeter = tnt.AverageValueMeter()
local checkpointLossMeter = tnt.AverageValueMeter()
local timeMeter = tnt.TimeMeter{unit = true}
local checkpointTimeMeter = tnt.TimeMeter{unit = true}

-- NOTE: Gradient normalization/averaging and hyper-parameters
-- Mini-batches have two dimensions: number of sequences (usually called batch
-- size) and number of tokens (i.e. time steps for recurrent models). Now, there
-- are two modes of normalizing gradients: by batch size only or by the total
-- number of non-padding tokens in the mini-batch (by batch size and time
-- steps). The second option can be activated with -timeavg. However, keep in
-- mind that the learning rate and clipping hyper-parameters have different
-- meanings for each mode:
--   * When normalizing by batch size only, the learning rate is specified per
--     time step and the clipping threshold is specified per sequence.
--   * When normalizing by total number of tokens, both learning rate and
--     clipping threshold are applied directly to the normalized gradients.
-- The first mode is implemented by not normalizing gradients at all, but rather
-- by dividing the learning rate by batch size and multiplying the clipping
-- factor by batch size. For higher-order methods like Adam that perform
-- normalizations to decouple the learning rate from the magnitude of gradients,
-- the learning rate is not divided by the batch size.
-- For models trained with bptt (i.e. recurrent decoders), normalizing
-- by batch size only tends to work a little better; for non-recurrent models
-- with -bptt 0, the second option is preferred.
local optalgConfig = {
    learningRate = config.lr,
    timeAverage = config.timeavg,
}
config.lrscale = 1     -- for logging
config.minlr = 1e-4    -- when to stop annealing

if config.optim == 'sgd' then
    optalgConfig.method = optim.sgd
    optalgConfig.momentum = config.momentum
    if not optalgConfig.timeAverage then
        optalgConfig.learningRate = optalgConfig.learningRate / config.batchsize
        config.lrscale = config.batchsize
        config.minlr = config.minlr / config.batchsize
    end
elseif config.optim == 'adam' then
    optalgConfig.method = optim.adam
    config.minlr = 1e-5
elseif config.optim == 'nag' then
    optalgConfig.method = require('fairseq.optim.nag')
    optalgConfig.momentum = config.momentum
    config.minlr = 1e-5
else
    error('wrong optimization algorithm')
end

if config.model == 'npmt' then
    optalgConfig.prune_schedule = config.prune_schedule
    optalgConfig.prune_schedule_start_epoch = config.prune_schedule_start_epoch
    optalgConfig.schedule_max_segment_len = config.schedule_max_segment_len
    optalgConfig.max_segment_len = config.max_segment_len
end

local runGeneration, genconfig, gensets = nil, nil, {}
if not config.nobleu or config.validbleu then
    genconfig = pltablex.copy(config)
    genconfig.bptt = 0
    genconfig.beam = 1
    genconfig._maxlen = 200
    genconfig.batchsize = config.batchsize
    genconfig.ngpus = 1
    _, gensets = data.loadCorpus{
        config = genconfig,
        testsets = {'valid', 'test'},
    }
end
if config.validbleu then
    local model = engine:model()
    runGeneration = hooks.runGeneration{
        model = model,
        dict = config.dict,
        generate = function(model, sample)
            genconfig.minlen = 1
            genconfig.maxlen = genconfig._maxlen
            local searchf = {}
            if config.model ~= 'npmt' then
                searchf = search.greedy(model:type(), genconfig.dict, genconfig.maxlen)
            end
            return model:generate(genconfig, sample, searchf)
        end,
    }
end

-- Save engine state at checkpoints
local saveEpochState = function(state) end
local epochStatePath = plpath.join(config.savedir, 'state_epoch%d.th7')

local saveLastState = function(state) end
local lastStatePath = plpath.join(config.savedir, 'state_last.th7')
if not config.nosave then
    saveEpochState = hooks.saveStateHook(engine, epochStatePath)
    saveLastState = hooks.saveStateHook(engine, lastStatePath)
end

-- Setup engine hooks
engine.hooks.onStart = function(state)
    if not state.checkpoint then
        state.checkpoint = 0
    end
end

engine.hooks.onStartEpoch = hooks.shuffleData(seed)
engine.hooks.onJumpToEpoch = hooks.shuffleData(seed)

local annealing = (config.optim == 'sgd' or config.optim == 'nag' or config.optim == 'adam')
local onCheckpoint = hooks.call{
    function(state)
        state.checkpoint = state.checkpoint + 1
    end,
    hooks.onCheckpoint{
        engine = engine,
        config = config,
        lossMeter = checkpointLossMeter,
        timeMeter = checkpointTimeMeter,
        runTest = hooks.runTest(engine),
        testsets = {valid = corpus.valid, test = corpus.test},
        runGeneration = runGeneration,
        gensets = {valid = gensets.valid},
        annealing = annealing,
        earlyStopping = (not annealing and not config.noearlystop),
    },
    function(state)
        checkpointLossMeter:reset()
        checkpointTimeMeter:reset()
        lossMeter:reset()
        timeMeter:reset()

        engine:training()
    end,
    saveEpochState,
    saveLastState,
}

engine.hooks.onUpdate = hooks.call{
    hooks.updateMeters{
        lossMeter = lossMeter,
        timeMeter = timeMeter,
    },
    hooks.updateMeters{
        lossMeter = checkpointLossMeter,
        timeMeter = checkpointTimeMeter,
    },
    function(state)
        if timeMeter.n == config.log_interval then
            local loss = lossMeter:value()
            local ppl = math.pow(2, loss / math.log(2))
            local elapsed = timeMeter.n * timeMeter:value()
            local statsstr = string.format(
                '| epoch %03d | %07d updates | words/s %7d' ..
                '| trainloss %8.2f | train ppl %8.2f',
                state.epoch, state.t, lossMeter.n / elapsed, loss, ppl)
            if state.dictstats then
                statsstr = statsstr .. string.format(
                    ' | avg_dict_size %.2f',
                    state.dictstats.size / state.dictstats.n)
            end
            print(statsstr)
            io.stdout:flush()
            timeMeter:reset()
            lossMeter:reset()
        end

        if config.save_interval > 0 and
            state.epoch_t % config.save_interval == 0 then
            saveLastState(state)
        end
    end
}

engine.hooks.onEndEpoch = onCheckpoint
engine.hooks.onEnd = saveLastState
engine.hooks.onSample = hooks.computeSampleStats(config.dict)

if plpath.isfile(lastStatePath) and not config.nosave then
    print('| Found existing state, attempting to resume training')
    engine.hooks.onJumpToSample = function(state)
        -- Jumping to a sample can be time-consuming. If, for some reason, you
        -- find yourself frequently resuming from a saved state, increase
        -- -ndatathreads to speed this up -- but keep in mind that this makes
        -- the sample order non-deterministic.
        if state.jumped % config.log_interval == 0 then
            print(string.format(
                '| epoch %03d | %07d updates | %07d epoch updates | %07d replayed',
                state.epoch, state.t, state.epoch_t, state.jumped))
        end
    end

    -- Support modifying the maxepoch setting during resume
    engine.hooks.onResume = function(state)
        state.maxepoch = config.maxepoch
        state.maxbatch = config.maxbatch
        state.group_size = config.group_size
        state.optconfig = optalgConfig
        if config.reset_lrate then
            print('Reset lr to ', config.lr)
            state.optconfig.learningRate = config.lr
        end
    end

    engine:resume{
        path = lastStatePath,
        iterator = corpus.train,
    }
else
    engine:train{
        iterator = corpus.train,
        optconfig = optalgConfig,
        maxepoch = config.maxepoch,
        clip = config.clip,
    }
end

local function runFinalEval()
    -- TODO
    local checkpoint_paths = {'model_best.th7', 'model_bestbleu.th7'}
    for icheckpoint_path = 1, #checkpoint_paths  do
        print('checkpoint', checkpoint_paths[icheckpoint_path])
        local path = plpath.join(config.savedir, checkpoint_paths[icheckpoint_path])
        local best_model = torch.load(path)

        genconfig.batchsize = 1
        genconfig.minlen = 1
        genconfig.maxlen = genconfig._maxlen

        for _, beam in ipairs({1}) do
            genconfig.beam = beam
            if not config.notext then
                genconfig.outfile = plpath.join(
                    config.savedir, string.format('gen-b%02d.txt', beam)
                )
            end
            local searchf = {}
            if config.model ~= 'npmt' then
                searchf = search.beam{
                ttype = best_model:type(),
                dict = genconfig.dict,
                srcdict = genconfig.srcdict,
                beam = genconfig.beam
                }
            end
            local _, result = hooks.runGeneration{
                model = best_model,
                dict = genconfig.dict,
                generate = function(model, sample)
                    return model:generate(genconfig, sample, searchf)
                end,
                outfile = genconfig.outfile,
                srcdict = config.srcdict,
            }(gensets.test)
            print(string.format('| Test with beam=%d: %s', beam, result))
            io.stdout:flush()
        end
    end
end



if not config.nobleu and not config.nosave then
    engine:executeAll(
        function(id)
            _G.model:network():clearState()
            _G.model = nil
            _G.params = nil
            _G.gradparams = nil
            collectgarbage()
        end
    )
    corpus, engine = nil, nil
    collectgarbage()
    runFinalEval()
end
