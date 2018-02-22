-- Copyright (c) Microsoft Corporation. All rights reserved.
-- Licensed under the MIT License.
--
--[[
--
-- NPMT model
-- it should actually be a criterion, but since it itself has
-- parameters, we still treat it as a module.
--
--]]
--
local NPMT, parent = torch.class('nn.NPMT', 'nn.Container')

function torch.tab_to_ngram(tab)
  local ngram = {}
  for i = 1, #tab do
    ngram[i] = "<s> " .. table.concat(tab[i], " ")
  end
  return ngram
end

function NPMT:__init(configs)
  parent.__init(self)
  self.use_cuda = configs.use_cuda or true
  self.num_thread = self.num_thread or 5
  self.max_segment_len = configs.max_segment_len or 5
  self.ngpus = configs.ngpus or 1
  print("npmt is running on ", self.ngpus, " GPUs")

  self.use_cimpl = configs.use_cimpl or true
  self.use_accel = configs.use_accel or false

  if self.use_cimpl then
    require "libdp_lib"
  end
  if self.use_accel then -- TODO
    print('Use CUDA accel')
    require "libcompute_logpy_lib"
  end

  self.vocab_size = configs.target_vocab_size or configs.dict:size() + 2
  self.embedding_size = configs.nembed or 256
  self.dec_unit_size = configs.dec_unit_size or configs.nhid or 256
  self.num_layers = configs.num_dec_layers or configs.nlayer or 1
  self.grad_check = configs.grad_check or false
  self.rnn_mode = configs.npmt_rnn_mode or configs.rnn_mode or "LSTM"
  self.nnlm_rnn_mode = configs.npmt_nnlm_rnn_mode or self.rnn_mode
  if self.grad_check then
    require('rnn')
    self.start_symbol = configs.start_symbol or self.vocab_size
  else
    self.start_symbol = configs.start_symbol or self.vocab_size - 1
  end

  self.end_segment_symbol = configs.end_segment_symbol or self.vocab_size
  self.pad_index = configs.dict.pad_index
  self.use_nnlm = configs.use_nnlm or false
  self.group_size = configs.group_size or 512

  self.report_time = configs.report_time or false
  self.precompute_gradInput = true
  self.lm_concat = configs.lm_concat or false
  self.dropout = configs.npmt_dropout or configs.dropout or 0
  self.nnlm_dropout = (configs.nnlm_dropout and configs.nnlm_dropout > 0 and configs.nnlm_dropout) or self.dropout or 0

  if self.dropout > 0 then
    print("npmt is using dropout ", self.dropout)
  end

  self.seq = nn.Sequential()
  if self.use_cuda then
    require "cudnn"
    local cudnn_mode = string.format("CUDNN_%s", self.rnn_mode)
    local rnn = nn.mRNN(self.embedding_size, self.dec_unit_size, false, cudnn_mode, false, configs.use_resnet_dec)
    self.seq:add(rnn)
    if self.dropout > 0 then
      self.seq:add(nn.Dropout(self.dropout))
    end
    for i = 2, self.num_layers do
      local rnn = nn.mRNN(self.dec_unit_size, self.dec_unit_size, false, cudnn_mode, false, configs.use_resnet_dec)
      self.seq:add(rnn)
      if self.dropout > 0 then
        self.seq:add(nn.Dropout(self.dropout))
      end
    end
  else
    local rnn_class
    if self.rnn_mode == "GRU" then
      rnn_class = nn.SeqGRU
    else
      rnn_class = nn.SeqLSTM
    end
    local rnn = rnn_class(self.embedding_size, self.dec_unit_size)
    rnn.batchfirst = true
    self.seq:add(rnn)
    for i = 2, self.num_layers do
      rnn = rnn_class(self.dec_unit_size, self.dec_unit_size)
      rnn.batchfirst = true
      self.seq:add(rnn)
    end
  end

  self.sub_outnet = nn.Sequential()
  self.sub_outnet:add(self.seq)
  self.sub_outnet:add(nn.Contiguous())
  self.sub_outnet:add(nn.View(-1, self.dec_unit_size))
  self.sub_outnet:add(nn.Linear(self.dec_unit_size, self.vocab_size))
  self.sub_outnet:add(nn.LogSoftMax())

  self.outnet = nn.Sequential()
  self.dict = nn.LookupTable(self.vocab_size, self.embedding_size)
  self.outnet:add(nn.ParallelTable():add(nn.Identity()):add(self.dict))
  self.outnet:add(self.sub_outnet)

  self:add(self.outnet)

  if self.use_nnlm then -- if we opt to use an additional language model for input
    self.nnlm = nn.Sequential()
    if configs.npmt_separate_embeddding then
      self.nnlm_dict = nn.LookupTable(self.vocab_size, self.embedding_size)
    else
      self.nnlm_dict = self.dict:clone("weight", "gradWeight") -- sharing the weights with self.dict
    end
    self.nnlm:add(self.nnlm_dict)
    self.nnlm_rnn = nn.Sequential()
    local nnlm_rnn_inst
    if self.use_cuda then
      local cudnn_mode = string.format("CUDNN_%s", self.nnlm_rnn_mode)
      nnlm_rnn_inst = nn.mRNN(self.embedding_size, self.dec_unit_size, false, cudnn_mode, false, configs.use_resnet_dec)
      self.nnlm_rnn:add(nnlm_rnn_inst)
      if self.nnlm_dropout> 0 then
        self.nnlm_rnn:add(nn.Dropout(self.nnlm_dropout))
      end
      for i = 2, self.num_layers do
        nnlm_rnn_inst = nn.mRNN(self.dec_unit_size, self.dec_unit_size, false, cudnn_mode, false, configs.use_resnet_dec)
        self.nnlm_rnn:add(nnlm_rnn_inst)
        if self.nnlm_dropout > 0 then
          self.nnlm_rnn:add(nn.Dropout(self.nnlm_dropout))
        end
      end
    else
      if self.rnn_mode == "GRU" then
        nnlm_rnn_inst = nn.SeqGRU(self.embedding_size, self.dec_unit_size)
      else
        nnlm_rnn_inst = nn.SeqLSTM(self.embedding_size, self.dec_unit_size)
      end
      nnlm_rnn_inst.batchfirst = true
      self.nnlm_rnn:add(nnlm_rnn_inst)
      for i = 2, self.num_layers do
        if self.rnn_mode == "GRU" then
          nnlm_rnn_inst = nn.SeqGRU(self.dec_unit_size, self.dec_unit_size)
        else
          nnlm_rnn_inst = nn.SeqLSTM(self.dec_unit_size, self.dec_unit_size)
        end
        nnlm_rnn_inst.batchfirst = true
        self.nnlm_rnn:add(nnlm_rnn_inst)
      end
    end
    self.nnlm:add(self.nnlm_rnn)
    self:add(self.nnlm)
    if self.lm_concat then
      self.lm_concat_proj = nn.Linear(self.dec_unit_size*2, self.dec_unit_size, false)
      self:add(lm_concat_proj)
    end
  end

  self.logpy = torch.Tensor()
  self.alpha = torch.Tensor()
  self.beta = torch.Tensor()
  self.logpy_per_data = torch.Tensor()
  self.seg_weight = torch.Tensor()
  self.seg_weight_cum = torch.Tensor()

  if self.use_cuda then
    cudnn.convert(self.outnet, cudnn)
    self:cuda()
  end
end

function NPMT:get_jstart_range(t, T1, minT2, maxT2)
  return math.max(1, minT2 - (T1-t+1)* self.batch_max_segment_len + 1), math.min(maxT2+1, (t-1) * self.batch_max_segment_len + 1)
end

function NPMT:compute_logpy(hidden_inputs, xlength, yref, ylength, batch_size, T1, T2)
  self.outnet:evaluate() --- set outnet in evalate mode
  self.logpy:resize(batch_size, T1, T2+1, self.batch_max_segment_len+1):fill(-torch.loginf())
--  self.logpy_c:resize(batch_size, T1, T2+1, self.batch_max_segment_len+1):fill(-torch.loginf())
  -- for word-based, each word is padded with zero (so that we can --
  -- easily know how long each word is), then in the following code,
  -- we will turn this into a proper sequence and padded with
  -- end-symbol
  --
  -- for letter-based, the entire sequence is padded with end-symbol

  local start_vector = yref.new(batch_size,1):fill(self.start_symbol)
  if torch.type(start_vector) ~= "torch.CudaTensor" then
    start_vector = start_vector:long()
  end

  if self.use_nnlm then
    self.nnlm_input = torch.cat(start_vector, yref)
    self.nnlm_output = self.nnlm:forward(self.nnlm_input)
  end

  local y_input
  local minT2 = ylength:min()

  local schedule = {}
  for t = 1, T1 do
    local jstart_l, jstart_u = self:get_jstart_range(t, T1, minT2, T2)
    for j_start = jstart_l, jstart_u do
      local j_len = math.min(self.batch_max_segment_len, T2-j_start+1)
      local j_end = j_start + j_len - 1
      table.insert(schedule, {t, j_start, j_len, j_end})
    end
  end
  if #schedule == 0 then
    return nil
  end
  local _, schedule_order = torch.sort(torch.Tensor(schedule)[{{}, 3}])
  local sorted_schedule = {}
  for si = 1, #schedule do
    table.insert(sorted_schedule, schedule[schedule_order[si]])
  end

  self.sorted_schedule = sorted_schedule
  local sorted_schedule_tensor = torch.CudaIntTensor(sorted_schedule) -- for compute_logpy_post use

--  print(os.clock(), "forward", self.sorted_schedule[{1,1}], T1, hidden_inputs:size(2))

  local concat_inputs = torch.Tensor()
  local concat_hts = torch.Tensor()
  if self.use_cuda then
    concat_inputs = concat_inputs:cuda()
    concat_hts = concat_hts:cuda()
  end
  self.group_size = math.max(self.group_size, batch_size)

  concat_inputs:resize(self.group_size, self.batch_max_segment_len + 1)
  concat_hts:resize(self.group_size, self.dec_unit_size)

  local si = 1
  while si <= #sorted_schedule do
    local si_next = math.min(si + math.floor(self.group_size / batch_size) - 1, #sorted_schedule)
    local s = si_next - si + 1
    local max_jlen = sorted_schedule[si_next][3]

    local t_concatInputs = concat_inputs[{{1, s * batch_size}, {1, 1 + max_jlen}}]
    local t_concatHts = concat_hts[{{1, s * batch_size}, {}}]
    t_concatInputs:fill(self.end_segment_symbol)
    t_concatHts:zero()

    for ell = si, si_next do
      local t, j_start, j_len, j_end = unpack(sorted_schedule[ell])
      local low_idx, high_idx = (ell-si)*batch_size+1, (ell-si+1)*batch_size
      y_input = start_vector:clone()
      if j_len > 0 then
        y_input = torch.cat({y_input, yref[{{}, {j_start,j_end}}]})
      end
      local hidden_input = hidden_inputs[{{}, t, {}}]
      if self.use_nnlm then
        if self.lm_concat then
          local hidden_input_concat = torch.cat(hidden_input, self.nnlm_output[{{}, j_start, {}}], 2)
          hidden_input = self.lm_concat_proj:updateOutput(hidden_input_concat)
        else
          hidden_input = torch.add(hidden_input, self.nnlm_output[{{}, j_start, {}}])
        end
      end
      t_concatHts[{{low_idx, high_idx}, {}}]:copy(hidden_input)
      t_concatInputs[{{low_idx, high_idx}, {1, y_input:size(2)}}]:copy(y_input)
    end

    local t_prob_all = self.outnet:updateOutput({t_concatHts, t_concatInputs}):view(s*batch_size, max_jlen+1, self.vocab_size)

    if self.use_accel then
      compute_logpy_post( t_prob_all, yref, ylength, self.logpy, sorted_schedule_tensor,
        s, max_jlen, self.vocab_size, batch_size, self.batch_max_segment_len, T1, T2, si)
    else
      -- Torch version of compute_logpy_post
      local t_vec = t_prob_all.new(batch_size)
      local t_valid = t_prob_all.new(batch_size)
      for ell = si, si_next do
        local t, j_start, j_len, j_end = unpack(sorted_schedule[ell])
        local low_idx, high_idx = (ell-si)*batch_size+1, (ell-si+1)*batch_size
        local t_prob = t_prob_all[{{low_idx, high_idx}, {}, {}}]

        local t_vec_whole = nil
        if j_len > 0 then
          t_vec_whole = t_prob[{{},{1, j_len},{}}]:gather(3, yref[{{},{j_start, j_end}}]:contiguous():view(batch_size, j_len, 1)):view(batch_size, j_len)
        end

        t_valid:copy(ylength:ge(j_start-1)) -- a 0/1 vector of length batch_size
        self.logpy[{{},t,j_start,1}]:add(torch.cmul(t_valid, torch.loginf() + t_prob[{{},1,self.end_segment_symbol}]))

        t_vec:zero()
        for j = j_start, j_end do --- this implies j_end >= j_start (when j=j_start-1, it means an empty segment)
          t_valid:copy(ylength:ge(j)) -- a 0/1 vector of length batch_size
          -- Use gather to fetch the corresponding values in the yref
          t_vec:add(t_vec_whole[{{}, j-j_start+1}])
          -- when j = j_start-1, this j-j_start+2 is 1, which is the first index, in NPMT,
          -- index 1 is for empty segment (segment length 0) while in segment.lua, index 1 is for
          -- segment length 1. So they differ by shifting 1 index.
          -- If non-empty, add end_symbol + t_vec; else add end_symbol
          self.logpy[{{},t,j_start,j-j_start+2}]:add(
            torch.cmul(t_valid, torch.loginf() + t_vec + t_prob[{{},j-j_start+2,self.end_segment_symbol}]))
        end
      end
    end
    si = si_next + 1
  end
  -- For debug use. Need to declare and use logpy_c
--  print(torch.all(torch.eq(self.logpy_c, self.logpy)))
--  io.write("finall: Press <Enter> to continue...")
--  io.read()
end

function NPMT:print_best_path(xlength_, yref_, ylength_, vocab)
  assert(self.alpha:size(1) == 1) -- only work for batch size 1
  local T1 = xlength_[1]
  local T2 = ylength_[1]
  local yref = yref_[1]
  local logpy = self.logpy[{1, {}, {}, {}}]
  local alpha = logpy.new(T1+1, T2+1)
  local prev = logpy.new(T1+1, T2+1):fill(-1)
  alpha:fill(-torch.loginf())
  alpha[{1,1}] = 0
  for t = 1, T1 do
    for j = 0, T2 do
      local j_low = math.max(1, j-self.batch_max_segment_len+1)
      for j_start = j_low, j+1 do
        local logprob = alpha[{t, j_start}] + logpy[{t, j_start, j-j_start+2}]
        if logprob > alpha[{t+1, j+1}] then
          alpha[{t+1, j+1}] = logprob
          prev[{t, j+1}] = j_start-1
        end
      end
    end
  end
  local j = T2
  local out_str = "|"
  for t = T1, 1, -1 do
    local prev_j = prev[{t, j+1}]
    for k = j, prev_j+1, -1 do
      out_str = vocab[yref[k]] .. out_str
    end
    if j > prev_j then
      out_str = "|" .. out_str
    end
    j = prev_j
  end
  print("best path: ", out_str)
  return out_str
end

function NPMT:alpha_and_beta(xlength, ylength, batch_size, T1, T2)
  self.alpha:resize(batch_size, T1+1, T2+1):fill(-torch.loginf())
  self.beta:resize(batch_size, T1+1, T2+1):fill(-torch.loginf())
  self.seg_weight:resizeAs(self.logpy):fill(-torch.loginf())

  if self.use_cimpl then
    self.logpy = self.logpy:double()
    self.alpha = self.alpha:double()
    self.beta = self.beta:double()
    ylength = ylength:double()
    xlength = xlength:double()
    self.seg_weight = self.seg_weight:double()

    c_sample_dp(
        batch_size,
        T1,
        T2,
        self.batch_max_segment_len,
        self.num_thread,
        tonumber(torch.data(self.logpy, true)),
        tonumber(torch.data(self.alpha, true)),
        tonumber(torch.data(self.beta, true)),
        tonumber(torch.data(self.seg_weight, true)),
        tonumber(torch.data(ylength, true)),
        tonumber(torch.data(xlength, true)))

    if (self.use_cuda) then
      self.logpy = self.logpy:cuda()
      self.alpha = self.alpha:cuda()
      self.beta = self.beta:cuda()
      self.seg_weight = self.seg_weight:cuda()
      ylength = ylength:cuda()
      xlength = xlength:cuda()
    else
      ylength = ylength:long()
      xlength = xlength:long()
    end
  else
    --- not use c implementation ---
    self.alpha[{{}, 1, 1}]:zero()
    for t = 1, T1 do
      for j = 0, T2 do
        local j_low = math.max(1, j-self.batch_max_segment_len+1)
        for j_start = j_low, j+1 do
          local logprob = self.alpha[{{}, t, j_start}] + self.logpy[{{}, t, j_start, j-j_start+2}]
          self.alpha[{{}, t+1, j+1}] = torch.logadd(self.alpha[{{}, t+1, j+1}], logprob)
        end
      end
    end
    for i = 1, batch_size do
      self.beta[{i, xlength[i]+1, ylength[i]+1}] = 0
    end
    for t = T1-1, 0, -1 do
      for j = 0, T2 do
        for j_end = j, math.min(T2, j + self.batch_max_segment_len) do
          local logprob = self.beta[{{}, t+2, j_end+1}] + self.logpy[{{}, t+1, j+1, j_end-j+1}]
          self.beta[{{}, t+1, j+1}] = torch.logadd(self.beta[{{}, t+1, j+1}], logprob)
        end
      end
    end

    local minT2 = ylength:min()
    for t = 1, T1 do
      local jstart_l, jstart_u = self:get_jstart_range(t, T1, minT2, T2)
      for j_start = jstart_l, jstart_u do
        local j_len = math.min(self.batch_max_segment_len, T2-j_start+1)
        local j_end = j_start + j_len - 1
        for j = j_start-1, j_end do
          self.seg_weight[{{}, t, j_start, j-j_start+2}] = self.logpy[{{}, t, j_start, j-j_start+2}]
                                                         + self.alpha[{{}, t, j_start}]
                                                         + self.beta[{{}, t+1, j+1}]
        end
      end
    end
  end

  self.logpy_per_data = self.beta[{{}, 1, 1}]:clone()
  if self.report_time then
    local logpy_per_data_alpha = self.alpha.new(batch_size)
    for i = 1, batch_size do
      logpy_per_data_alpha[i] = self.alpha[{i, xlength[i]+1, ylength[i]+1}]
    end
    print(string.format("%.25f", torch.sum(self.logpy_per_data) - torch.sum(logpy_per_data_alpha)))
    print(torch.sum(self.logpy_per_data))
  end

  self.seg_weight:add(-self.logpy_per_data:view(batch_size, 1, 1, 1):repeatTensor(1, T1, T2+1, self.batch_max_segment_len+1))
  self.seg_weight_cum:resizeAs(self.seg_weight):fill(-torch.loginf())

  if self.use_cimpl then
    self.seg_weight_cum:copy(self.seg_weight)
    self.seg_weight_cum = self.seg_weight_cum:double()
    ylength = ylength:double()
    c_reverse_log_cumsum(
        batch_size,
        T1,
        T2,
        self.batch_max_segment_len,
        self.num_thread,
        tonumber(torch.data(self.seg_weight_cum, true)),
        tonumber(torch.data(ylength, true)))
    if (self.use_cuda) then
      ylength= ylength:cuda()
      self.seg_weight_cum = self.seg_weight_cum:cuda()
    else
      ylength = ylength:long()
    end
    self.seg_weight:exp() -- make it actual weight
    self.seg_weight_cum:exp()
  else
    self.seg_weight:exp() -- make it actual weight
    self.seg_weight_cum:copy(self.seg_weight)
    self.seg_weight_cum = self.seg_weight_cum:index(4, torch.linspace(self.batch_max_segment_len+1, 1, self.batch_max_segment_len+1):long())
                                             :cumsum(4)
                                             :index(4, torch.linspace(self.batch_max_segment_len+1, 1, self.batch_max_segment_len+1):long())
  end
end

function NPMT:compute_gradients(hidden_inputs, xlength, yref, ylength, batch_size, T1, T2)
  self.outnet:training() --- set outnet in training mode
  local grad_hidden_inputs = hidden_inputs.new(hidden_inputs:size()):zero()

  local start_vector = yref.new(batch_size,1):fill(self.start_symbol)
  if torch.type(start_vector) ~= "torch.CudaTensor" then
    start_vector = start_vector:long()
  end

  local nnlm_gradOutput
  if self.use_nnlm then
    nnlm_gradOutput = self.nnlm_output.new(self.nnlm_output:size()):zero()
  end

  local sorted_schedule = self.sorted_schedule -- copy from forward
  assert(#sorted_schedule > 0)
--  print(os.clock(), "backward", self.sorted_schedule[{1,1}], T1, hidden_inputs:size(2))

  local concat_inputs = torch.Tensor()
  local concat_hts = torch.Tensor()
  local gradOutput = torch.Tensor()
  if (self.use_cuda) then
    concat_inputs = concat_inputs:cuda()
    concat_hts = concat_hts:cuda()
    gradOutput = gradOutput:cuda()
  end

  concat_inputs:resize(self.group_size, self.batch_max_segment_len + 1)
  concat_hts:resize(self.group_size, self.dec_unit_size)
  gradOutput:resize(self.group_size, self.batch_max_segment_len + 1, self.vocab_size)

  local grad_scale = -1.0 / (yref:size(1) * self.ngpus)
  local y_input
  local si = 1

  local skip_sample = false
  for si = 1, #sorted_schedule do
    if sorted_schedule[si][1] > hidden_inputs:size(2) then
      skip_sample = true
    end
  end
  if skip_sample then
    print('skip')
  else
    while si <= #sorted_schedule do
      local si_next = math.min(si + math.floor(self.group_size / batch_size) - 1, #sorted_schedule)
      local s = si_next - si + 1
      local max_jlen = sorted_schedule[si_next][3]

      local t_concatInputs = concat_inputs[{{1, s * batch_size}, {1, max_jlen + 1}}]
      local t_concatHts = concat_hts[{{1, s * batch_size}, {}}]
      local t_gradOutput = gradOutput[{{1, s * batch_size}, {1, max_jlen +1}, {}}]
      t_concatInputs:fill(self.end_segment_symbol)
      t_concatHts:zero()
      t_gradOutput:zero()

      for ell = si, si_next do
        local t, j_start, j_len, j_end = unpack(sorted_schedule[ell])
        local low_idx, high_idx = (ell-si)*batch_size+1, (ell-si+1)*batch_size
        y_input = start_vector:clone()
        if j_end >= j_start then
          y_input = torch.cat({y_input, yref[{{}, {j_start,j_end}}]})
        end
--        if t > hidden_inputs:size(2) then
--          print("xlength", xlength)
--          print("ylength", ylength)
--          print("ylength", yref)
--          print("size", hidden_inputs:size())
--          print("t", t, j_start, j_len, j_end, T1)
--        end
        local hidden_input = hidden_inputs[{{}, t, {}}]
        if self.use_nnlm then
          if self.lm_concat then
            local hidden_input_concat = torch.cat(hidden_input, self.nnlm_output[{{}, j_start, {}}], 2)
            hidden_input = self.lm_concat_proj:updateOutput(hidden_input_concat)
          else
            hidden_input = torch.add(hidden_input, self.nnlm_output[{{}, j_start, {}}])
          end
        end
        t_concatHts[{{low_idx, high_idx}, {}}]:copy(hidden_input)
        t_concatInputs[{{low_idx, high_idx}, {1, y_input:size(2)}}]:copy(y_input)
        if j_len > 0 then
          local yweight = self.seg_weight_cum[{{}, t, j_start, {2, j_len+1}}]:contiguous()
          local ysnipt = yref[{{}, {j_start, j_end}}]:contiguous()
          -- Use scatter to put batch of y batch to corresponding place
          t_gradOutput[{{low_idx, high_idx}, {1, j_len}, {}}]:scatter(3, ysnipt:view(batch_size, j_len, 1), yweight:view(batch_size, j_len, 1))
        end
        t_gradOutput[{{low_idx, high_idx}, {1,j_len+1}, self.end_segment_symbol}]:copy(self.seg_weight[{{}, t, j_start, {1,j_len+1}}])
      end

      t_gradOutput:mul(grad_scale)
      local reshaped_t_gradOutput = t_gradOutput:reshape(s*batch_size*(max_jlen+1), self.vocab_size)
      self.outnet:forward({t_concatHts, t_concatInputs})
      self.outnet:backward({t_concatHts, t_concatInputs}, reshaped_t_gradOutput)
      reshaped_t_gradOutput:set()


      for ell = si, si_next do
        local t, j_start, j_len, j_end = unpack(sorted_schedule[ell])
        local t_valid = gradOutput.new(batch_size):zero()
        t_valid:copy(xlength:ge(t))
        local low_idx, high_idx =  (ell-si)*batch_size+1, (ell-si+1)*batch_size
        local grad_input = torch.cmul(self.outnet.gradInput[1][{{low_idx, high_idx}, {}}],
                                      t_valid:view(batch_size, 1):expand(batch_size, hidden_inputs:size(3)))
        if self.nnlm then
          if self.lm_concat then
            local hidden_input_concat = torch.cat(hidden_inputs[{{}, t, {}}], self.nnlm_output[{{}, j_start, {}}], 2)
            local concat_grad_input = self.lm_concat_proj:backward(hidden_input_concat, grad_input)
            grad_hidden_inputs[{{},t,{}}]:add(concat_grad_input[{{}, {1, self.dec_unit_size}}])
            nnlm_gradOutput[{{}, j_start, {}}]:add(concat_grad_input[{{}, {self.dec_unit_size + 1, 2*self.dec_unit_size}}])
          else
            nnlm_gradOutput[{{}, j_start, {}}]:add(grad_input)
            grad_hidden_inputs[{{},t,{}}]:add(grad_input)
          end
        else
          grad_hidden_inputs[{{},t,{}}]:add(grad_input)
        end
      end
      si = si_next + 1
    end
  end
  if self.use_nnlm then
    self.nnlm:backward(self.nnlm_input, nnlm_gradOutput)
  end

  if (self.report_time) then
      print('     compute time => ', os.clock() - t_clock)
  end

  self.gradInput = {grad_hidden_inputs,
                    xlength.new(xlength:size()):zero(),
                    yref.new(yref:size()):zero(),
                    ylength.new(ylength:size()):zero()}
end

function NPMT:forward_and_backward(input)
  local t_clock = nil
  if self.report_time then
    t_clock = os.clock()
  end
  local hidden_inputs, xlength, yref, ylength = unpack(input)
  --  hidden_inputs: [torch.CudaTensor of size batch_size, T1, hidden]
  --  yref: [torch.CudaTensor of size batch_size, T2]
  --  xlength: [torch.CudaTensor of size batch_size]
  --  ylength: [torch.CudaTensor of size batch_size]

  local batch_size = hidden_inputs:size(1)
  local T1, T2 = hidden_inputs:size(2), yref:size(2)
  self.batch_max_segment_len = math.min(self.max_segment_len, T2)

  assert(hidden_inputs:dim() == 3)
  if torch.type(yref) ~= "torch.CudaTensor" then
    xlength = xlength:long()
    yref = yref:long()
    ylength = ylength:long()
  else
    xlength = xlength:cuda()
    yref = yref:cuda()
    ylength = ylength:cuda()
  end

  --Initialization
  self.output = nil
  if self.precompute_gradInput then
    self.gradInput = {}
  end

  if (self.report_time) then
    print(' Initialization time => ', os.clock() - t_clock)
  end

  -- Step 1: compute log p(y_{j1:j2}|h_t)
  if (self.report_time) then
    t_clock = os.clock()
  end

  self:compute_logpy(hidden_inputs, xlength, yref, ylength, batch_size, T1, T2)

  if (self.report_time) then
    print(' Time for phase 1 ==> ', os.clock() - t_clock)
  end

  -- Step 2: sum over the probs
  if (self.report_time) then
    t_clock = os.clock()
  end

  self:alpha_and_beta(xlength, ylength, batch_size, T1, T2)
  self.output = self.logpy_per_data.new(1):fill(-torch.sum(self.logpy_per_data))

  if (self.report_time) then
    print(' Time for phase 2 ==> ', os.clock() - t_clock)
  end

  -- Step 3: compute gradients
  if (self.report_time) then
    t_clock = os.clock()
  end

  if (self.precompute_gradInput) then
    self:compute_gradients(hidden_inputs, xlength, yref, ylength, batch_size, T1, T2)
  end

  if (self.report_time) then
    print(' Time for phase 3 ==> ', os.clock() - t_clock)
  end
end

function NPMT:forward(input)
  self:forward_and_backward(input)
  return self.output
end

function NPMT:backward(input, gradOutput, scale)
  assert(self.precompute_gradInput)
  return self.gradInput
end

function NPMT:updateOutput(input)
  return self:forward(input)
end

function NPMT:updateGradInput(input, gradOutput)
  if self.precompute_gradInput then
    self:backward(input, gradOutput, 1.0)
  end
  return self.gradInput
end

function NPMT:accGradParameters(input, gradOutput, scale)
  if self.precompute_gradInput then
    self:backward(input, gradOutput, scale)
  end
end

function NPMT:SelectRememberRNNStates(rnns, new_ht_idx)
  local ht
  if self.use_cuda then
    if self.rnn_mode == "LSTM" then
      ht = rnns[1].cellOutput:view(-1, self.dec_unit_size)
                             :index(1, torch.LongTensor(new_ht_idx))
                             :view(1, #new_ht_idx, self.dec_unit_size)
    else
      ht = rnns[1].hiddenOutput:view(-1, self.dec_unit_size)
                               :index(1, torch.LongTensor(new_ht_idx))
                               :view(1, #new_ht_idx, self.dec_unit_size)
    end
    for i = 1, #rnns do
      rnns[i].hiddenInput = rnns[i].hiddenOutput:view(-1, self.dec_unit_size)
                                                :index(1, torch.LongTensor(new_ht_idx))
                                                :view(1, #new_ht_idx, self.dec_unit_size)
      if self.rnn_mode == "LSTM" then
        rnns[i].cellInput = rnns[i].cellOutput:view(-1, self.dec_unit_size)
                                              :index(1, torch.LongTensor(new_ht_idx))
                                              :view(1, #new_ht_idx, self.dec_unit_size)
      end
    end
  else
    --TODO fix lstm on cpu
    ht = rnns[1]._output[1]:view(-1, self.dec_unit_size)
                           :index(1, torch.LongTensor(new_ht_idx))
                           :view(#new_ht_idx, self.dec_unit_size)
    for i = 1, #rnns do
      rnns[i].h0 = rnns[i]._output[1]:view(-1, self.dec_unit_size)
                                     :index(1, torch.LongTensor(new_ht_idx))
                                     :view(#new_ht_idx, self.dec_unit_size)
      if rnns[i].cell then
        rnns[i].c0 = rnns[i].cell[1]:view(-1, self.dec_unit_size)
                                    :index(1, torch.LongTensor(new_ht_idx))
                                    :view(#new_ht_idx, self.dec_unit_size)
      end
    end
  end
  return ht
end

function NPMT:rememberRNNStates(rnns)
  local ht
  if self.use_cuda then
    if self.rnn_mode == "LSTM" then
      ht = rnns[1].cellOutput:clone()
    else
      ht = rnns[1].hiddenOutput:clone()
    end
    for i = 1, #rnns do --remember old states
      rnns[i].hiddenInput = rnns[i].hiddenOutput:clone()
      if self.rnn_mode == "LSTM" then
        rnns[i].cellInput = rnns[i].cellOutput:clone()
      end
    end
  else
    --TODO fix lstm on cpu
    ht = rnns[1]._output[1]:clone()
    for i = 1, #rnns do --remember old states
      rnns[i].h0 = rnns[i]._output[1]:clone()
      if rnns[i].cell then
        rnns[i].c0 = rnns[i].cell[1]:clone()
      end
    end
  end
  return ht
end

function NPMT:GetNNLMRnns()
  local rnns, rnn_containers
  if self.use_cuda then
    rnns, rnn_containers = self.nnlm:findModules("cudnn.RNN")
  else
    if self.rnn_mode == "LSTM" then
      rnns, rnn_containers = self.nnlm:findModules("nn.SeqLSTM")
    elseif self.rnn_mode == "GRU" then
      rnns, rnn_containers = self.nnlm:findModules("nn.SeqGRU")
    else
      assert(false)
    end
  end
  return rnns
end

function NPMT:GetOutnetRnns()
  local rnns, rnn_containers
  if self.use_cuda then
    rnns, rnn_containers = self.outnet:findModules("cudnn.RNN")
  else
    if self.rnn_mode == "LSTM" then
      rnns, rnn_containers = self.outnet:findModules("nn.SeqLSTM")
    elseif self.rnn_mode == "GRU" then
      rnns, rnn_containers = self.outnet:findModules("nn.SeqGRU")
    else
      assert(false)
    end
  end
  return rnns
end

function NPMT:clearOutnetStates()
  local rnns = self:GetOutnetRnns()
  for i = 1, #rnns do
    rnns[i]:resetStates()
  end
end

function NPMT:resetRNNStates()
  local rnns = self:GetOutnetRnns()
  for i = 1, #rnns do
    rnns[i]:resetStates()
  end
  if self.use_nnlm then
    rnns = self:GetNNLMRnns()
    for i = 1, #rnns do
      rnns[i]:resetStates()
    end
  end
end

function NPMT:training()
  self.precompute_gradInput = true
  self:resetRNNStates()
  parent.training(self)
end

function NPMT:evaluate()
  self.precompute_gradInput = false
  self:resetRNNStates()
  parent.evaluate(self)
end

function NPMT:clearState()
  parent.clearState(self)
  self:resetRNNStates()
  self.logpy:set()
  self.alpha:set()
  self.beta:set()
  self.logpy_per_data:set()
  self.seg_weight:set()
  self.seg_weight_cum:set()
  self.output = 0.
  self.gradInput = {}
  self.sorted_schedule = {}
  if self.use_nnlm then
    if self.nnlm_output ~= nil then
        self.nnlm_output:set()
    end
    if self.nnlm_input ~= nil then
        self.nnlm_input:set()
    end
  end
end

function NPMT:predict(input, xlength, test_mode)
  -- TODO fixes
  local batch_size, T1 = input:size(1), input:size(2)
--  assert(batch_size == 1)
  local max_segment_len = self.max_segment_len
  local sts_input = torch.Tensor()--:fill(self.start_symbol)
  if (self.use_cuda) then
    sts_input = sts_input:cuda()
  end
  local start_symbol = self.start_symbol
  local rnns = self:GetOutnetRnns()

  local tab_output_seqs = {}
  local tab_output_probs = {}
  local test_segments = 0

  for b = 1, batch_size do
    local out_str = "|"
    local output_symbol = nil
    local output_seq = {}
    local output_probs = {}
    local num_segments = 0
    sts_input:resize(1, 1):fill(self.start_symbol)

    local nnlm_rnns, nnlm_output
    if self.use_nnlm then
       nnlm_rnns = self:GetNNLMRnns()
       nnlm_output = self.nnlm:updateOutput(sts_input):view(-1, self.dec_unit_size)
       self:rememberRNNStates(nnlm_rnns)
    end

    for t = 1, xlength[b] do
      sts_input[1][1] = start_symbol
      local ht = input[{{b},t,{}}]:clone()
      self:clearOutnetStates()
      if self.use_nnlm then
        if self.lm_concat then
          ht = self.lm_concat_proj:updateOutput(torch.cat(ht, nnlm_output, 2))
        else
          ht:add(nnlm_output)
        end
      end
      local new_segment = false
      for j = 1, max_segment_len do
        local output_prob = self.outnet:updateOutput({ht, sts_input}):view(-1) -- 1-dimensional vector of length V
        local max_prob, output_symbol = output_prob:view(-1):max(1)
        output_symbol = output_symbol:squeeze()
        if output_symbol == self.end_segment_symbol then
          break -- finished reading this segment
        else
          if not new_segment then
            num_segments = num_segments + 1
            new_segment = true
          end

          table.insert(output_seq, output_symbol)
          table.insert(output_probs, max_prob[1])
          if test_mode then
--            out_str = out_str .. " " .. output_symbol
              -- input time t, output corresponding segments
              out_str = out_str .. " " .. t .. ':' .. output_symbol
--            out_str = out_str .. " " .. vocab[output_symbol]
          end
          sts_input[1][1] = output_symbol
          ht = self:rememberRNNStates(rnns)
          if self.use_nnlm then
            nnlm_output = self.nnlm:updateOutput(sts_input):view(-1, self.dec_unit_size)
            self:rememberRNNStates(nnlm_rnns)
          end
        end
      end
      if new_segment and test_mode then
        out_str = out_str .. '|'
      end
    end

    if #output_seq == 0 then
      local eos_index = 3
      table.insert(output_seq, eos_index) -- eos
      table.insert(output_probs, 1)
    end
    self:clearState() -- don't leave things for next one example
    if test_mode then
      print("max decoding:", out_str)
    end

    table.insert(tab_output_seqs, torch.Tensor(output_seq))
    table.insert(tab_output_probs, output_probs)  -- place holder, dummy
    test_segments = test_segments + num_segments
  end
  tab_output_seqs = nn.FlattenTable():forward(tab_output_seqs)
  local output_count = 0
  for i = 1, #tab_output_seqs do
    output_count = output_count + tab_output_seqs[i]:nElement()
  end
  return tab_output_seqs, nn.FlattenTable():forward(tab_output_probs), output_count, test_segments
end

function NPMT:beam_search(input, xlength, configs)
  local configs = configs or {}
  local word_weight = configs.lenpen or configs.word_weight or 0
  local beam_size = configs.beam_size or configs.beam or 20

  local lm_weight = configs.lm_weight or 0
  local lm = configs.lm

  local batch_size, T1 = input:size(1), input:size(2)
  local rnns = self:GetOutnetRnns()

  local max_segment_len = self.max_segment_len
  local sts_input = torch.Tensor():cuda()

  local tab_output_seqs = {}
  local tab_output_probs = {}

  for b = 1, batch_size do
    sts_input:resize(1, 1):fill(self.start_symbol)

    local fin_trans = {}
    local fin_probs = {}

    local nnlm_rnns, nnlm_output
    if self.use_nnlm then
      nnlm_output = self.nnlm:updateOutput(sts_input):view(-1, self.dec_unit_size)
    end

    for t = 1, xlength[b] do
      local trans_t = {}
      local probs_t = {}
      local fin_trans_t = {}
      local fin_probs_t = {}

      local ht = input[{{b},t,{}}]:clone()
      if t > 1 then
        sts_input:resize(#fin_trans, 1):fill(self.start_symbol)
        for i = 1, #fin_trans do
          table.insert(trans_t, torch.copy_array(fin_trans[i]))
        end
        probs_t = torch.copy_array(fin_probs)
        ht = ht:repeatTensor(#fin_trans, 1)
      else
        table.insert(trans_t, {})
        table.insert(probs_t, 0.)
        sts_input:resize(1, 1):fill(self.start_symbol)
      end

      local ngrams 
      if lm_weight > 0 then
        ngrams = torch.tab_to_ngram(trans_t)
      end

      -- nnlm 1. dropout, 2. nnlm with increasing/decreasing weights
      if self.use_nnlm then
        if self.lm_concat then
          ht = self.lm_concat_proj:updateOutput(torch.cat(ht, nnlm_output, 2))
        elseif self.schedule_nnlm > 0 then
          ht = ht + self.schedule_nnlm * nnlm_output
          ht = self.lm_concat_proj:updateOutput(torch.cat(ht, nnlm_output, 2))
        else
          ht:add(nnlm_output)
        end
      end
      self:clearOutnetStates()
      local nsamples = beam_size
      for j = 1, max_segment_len + 1 do
        local output_prob = self.outnet:updateOutput({ht, sts_input})
        local new_trans_t = {}
        local new_probs_t = {}
        local new_ht_idx = {}
        if j == max_segment_len + 1 then -- we have to force to have the end_segment_symbol for each input
          probs_t = torch.Tensor(probs_t):cuda() + output_prob[{{}, self.end_segment_symbol}] -- + word_weight
          for k = 1, nsamples do
            table.insert(fin_trans_t, trans_t[k])
            table.insert(fin_probs_t, probs_t[k])
          end
        else
          probs_t = torch.Tensor(probs_t):cuda():view(-1, 1):repeatTensor(1, self.vocab_size) + output_prob -- + word_weight
          probs_t = probs_t:view(-1)
          local _, sorted_idx = torch.sort(probs_t, true)
          for k = 1, math.min(nsamples, sorted_idx:nElement()) do
            local tran_id = math.floor((sorted_idx[k]-1) / self.vocab_size) + 1
            local word_id = (sorted_idx[k]-1) % self.vocab_size + 1
            if word_id == self.end_segment_symbol then
              --- end symbol
              table.insert(fin_trans_t, trans_t[tran_id])
              table.insert(fin_probs_t, probs_t[sorted_idx[k]])
            else
              --- continue in the pool
              local new_tran = torch.copy_array(trans_t[tran_id])
              table.insert(new_tran, word_id)
              table.insert(new_trans_t, new_tran)

              local new_prob = probs_t[sorted_idx[k]] + word_weight
              if lm_weight > 0 then
                new_prob = new_prob + lm_weight * lookup_lm_prob(lm, ngrams[tran_id], tostring(word_id))[1]
              end
              table.insert(new_probs_t, new_prob)
              table.insert(new_ht_idx, tran_id)
            end
          end
        end
        trans_t = new_trans_t
        probs_t = new_probs_t
        nsamples = #trans_t
        if nsamples == 0 then
          break
        end
        if lm_weight > 0 then
          ngrams = torch.tab_to_ngram(trans_t)
        end
        sts_input:resize(#trans_t, 1):copy(torch.last_nelements(trans_t, 1, self.start_symbol))
        ht = self:SelectRememberRNNStates(rnns, new_ht_idx)
      end
      if t > 1 then
        --- merge same sequences
        local merge_fin_trans_t = {}
        merge_fin_trans_t[table.concat(fin_trans_t[1], '-')] = {fin_trans_t[1], fin_probs_t[1]}
        for i = 2, #fin_trans_t do
          local tran_str = table.concat(fin_trans_t[i], '-')
          if merge_fin_trans_t[tran_str] ~= nil then
            merge_fin_trans_t[tran_str][2] = torch.logadd(merge_fin_trans_t[tran_str][2], fin_probs_t[i])
          else
            merge_fin_trans_t[tran_str] = {fin_trans_t[i], fin_probs_t[i]}
          end
        end
        fin_trans_t = {}
        fin_probs_t = {}
        for key, value in pairs(merge_fin_trans_t) do
          table.insert(fin_trans_t, value[1])
          table.insert(fin_probs_t, value[2])
        end
      end
      fin_trans = fin_trans_t
      fin_probs = fin_probs_t

      if self.use_nnlm then
        nnlm_output:resize(#fin_trans, self.dec_unit_size):zero()
        local max_len = #(fin_trans[1])
        for i = 1, #fin_trans do
          max_len = math.max(max_len, #(fin_trans[i]))
        end
        local lm_input = nnlm_output.new(#fin_trans, max_len+1):fill(self.start_symbol)
        for i = 1, #fin_trans do
          if #fin_trans[i] > 0 then
            lm_input[{i, {2,#(fin_trans[i])+1}}]:copy(torch.Tensor(fin_trans[i]))
          end
        end
        local lm_output = self.nnlm:updateOutput(lm_input)
        for i = 1, #fin_trans do
          nnlm_output[{i, {}}]:copy(lm_output[{i, #(fin_trans[i])+1, {}}])
        end
      end
    end

    if configs.use_avg_prob then
      for i = 1, #fin_probs do
        fin_probs[i] = fin_probs[i] / (#fin_trans[i])
      end
    end

    local _, sorted_idx = torch.sort(torch.Tensor(fin_probs), true)

    local output_seqs = {}
    local output_probs = {}

    for i = 1, math.min(configs.beam_size, #fin_probs) do
      if #fin_trans[sorted_idx[i]] > 0 then
        table.insert(output_seqs, torch.Tensor(fin_trans[sorted_idx[i]]))
      else
        local eos_index = 3
        table.insert(output_seqs, torch.Tensor(1):fill(eos_index))
      end
      table.insert(output_probs, fin_probs[sorted_idx[i]])
    end
    self:clearState() -- don't leave things for next one example
    table.insert(tab_output_seqs, output_seqs)
    table.insert(tab_output_probs, output_probs)
  end

  return nn.FlattenTable():forward(tab_output_seqs), nn.FlattenTable():forward(tab_output_probs)

end
