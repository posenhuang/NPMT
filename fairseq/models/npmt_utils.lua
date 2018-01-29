-- Copyright (c) Microsoft Corporation. All rights reserved.
-- Licensed under the MIT License.
--
--[[
--
-- Auxiliary functions
--
--]]
--

-- log(ret) = log(a + b)
function torch.logadd(log_a, log_b)
  if (type(log_a) == 'number') then
    return math.max(log_a, log_b) + torch.log1p(torch.exp(-torch.abs(log_a - log_b)))
  else
    return torch.cmax(log_a, log_b) + torch.log1p(torch.exp(-torch.abs(log_a - log_b)))
  end
end

-- log(ret) = log(1-a)
function torch.log1sub(log_a)
  return torch.log1p(-torch.exp(log_a))
end

-- log(ret) = log(a - b)
function torch.logsub(log_a, log_b)
  assert(torch.all(torch.ge(log_a, log_b)))
  return log_a + torch.log1p(-torch.exp(log_b - log_a) + 1e-15)
end

-- log(ret) = log(+inf)
function torch.loginf()
  return 1000000
end

function torch.copy_array(tab)
  return torch.Tensor(tab):totable()
end

function torch.last_nelements(tab, n, pad)
  local n = n or 1
  local pad = pad or -1
  local out = torch.Tensor(#tab, n)
  for i = 1, #tab do
    for j = 1, n do
      out[{i,j}] = tab[i][#(tab[i]) - n + j] or pad
    end
  end
  return out
end

-- generate n random rules for a vocabulary of V (excluding 1 and V) and probabilities specified in probs
-- return: a table of size n, each being a table of two 1-dimensional tensors
function gen_random_rules(V, n, probs, use_ctc, longer_input)
    local lens_src
    if use_ctc then
      lens_src = torch.Tensor(n):fill(probs:nElement())
      assert(not longer_input, "use_ctc can not be used with longer input")  
    else
      if longer_input then
        lens_src = torch.Tensor(n):random(3, 15)
      else
        lens_src = torch.multinomial(probs, n, true) 
      end
    end
    local lens_trgt = torch.multinomial(probs, n, true)
    if longer_input then
      lens_trgt = torch.Tensor(n):random(1, 6)
    end
    local rules = {}
    for i = 1, n do
        local t_src = torch.Tensor(lens_src[i]):random(1, V)
        local t_trgt = torch.Tensor(lens_trgt[i]):random(1, V)
        table.insert(rules, {t_src, t_trgt})
    end
    return rules
end

-- generate n sequences each of length T, using those specified in rules
-- return three tables: input, output and markers, each has exactly n elements
function gen_random_data(rules, n, T, sort_output)
    if sort_output then
        print("--- the outputs are sorted ---")
    end
    local input = {}
    local output = {}
    local markers = {}
    
    for i = 1, n do
        local output_indices = {}
        local t_src = {}
        local t_trgt = {}
        local t_marker = {}
        local count = 0
        local random_T = torch.random(1, T)
        local rule_idx = {}
        for k = 1, random_T do
          local j = torch.random(1, #rules)
          table.insert(rule_idx, j)
        end
        rule_idx = torch.sort(torch.Tensor(rule_idx))
        for k = 1, random_T do
          local j = rule_idx[k] 
          table.insert(t_src, rules[j][1])
          table.insert(t_trgt, rules[j][2])
          count = count + rules[j][1]:nElement()
          table.insert(t_marker, count)
          table.insert(output_indices, j)
        end
        table.insert(input, torch.cat(t_src))
        table.insert(markers, torch.Tensor(t_marker))
        if sort_output then
            local t_trgt_sorted = {}
            _, indices = torch.sort(torch.Tensor(output_indices))
            for j = 1, indices:nElement() do
              table.insert(t_trgt_sorted, t_trgt[indices[j]])   
            end
            table.insert(output, torch.cat(t_trgt_sorted))
        else
            table.insert(output, torch.cat(t_trgt))
        end
    end
    return input, output, markers
end

function toy_prepare_minibatch(input_table, output_table, markers_table, p, batch_size)

    local T1 = input_table[p]:nElement()
    local T2 = 0
    local input = torch.Tensor(batch_size, T1):fill(2)
    local markers = {}
    local ylength = torch.Tensor(batch_size)
    for i = 1, batch_size do
        input[{i,{}}]:copy(input_table[p+i-1])
        table.insert(markers, markers_table[p+i-1])
        ylength[i] = output_table[p+i-1]:nElement()
        if (ylength[i] > T2) then
            T2 = ylength[i]
        end
    end
    local yref = torch.Tensor(batch_size, T2):fill(2)
    for i = 1, batch_size do
        yref[{i,{1,ylength[i]}}]:copy(output_table[p+i-1])
    end

    return input, yref, ylength, markers
end

function toy_evaluate_zeta_loss(logprob, markers)

    local loss = 0
    local T1 = logprob:size(2)
    for i = 1, #markers do
        loss = loss + torch.sum(logprob[{i,{}}]:gather(1, markers[i]:long()))
    end

    return loss

end

function get_toy_data(V, sort_output, use_ctc, longer_input)
  local n_train = 16384
  local n_test = 128
  local m = 100
  local T1 = 6
  local V = V

  -- generate data
  local rules = gen_random_rules(V, m, torch.Tensor({1/3,1/3,1/3}), false, longer_input)
  local input_train, output_train, markers_train = gen_random_data(rules, n_train, T1, sort_output)
  local input_test, output_test, markers_test = gen_random_data(rules, n_test, T1, sort_output)
  return {rules, 
          input_train, 
          output_train, 
          markers_train,
          input_test, 
          output_test, 
          markers_test}
end

function sort_data_by_length(input, output, max_sen_len)
  max_sen_len = max_sen_len or nil
  local all_lengths = {}
  for i = 1, #input do
    table.insert(all_lengths, input[i]:nElement())
  end
  local _, sorted_idx = torch.sort(torch.Tensor(all_lengths))
  local sorted_input = {}
  local sorted_output = {}
  for i = 1, #input do
    local j = sorted_idx[i]
    if not max_sen_len then
      table.insert(sorted_input, input[j])
      table.insert(sorted_output, output[j])
    elseif input[j]:nElement() <= max_sen_len then
      table.insert(sorted_input, input[j])
      table.insert(sorted_output, output[j])
    end
  end
  return sorted_input, sorted_output
end


local log0 = -1000000.
function prepare_minibatch_3d(input, output, s, t, params)
    local y_vocab = params.target_vocab_size

    local input_max_len = 0
    local input_feature_dim = 123
    local output_max_len = 0
    local batch_size = t - s + 1
    for i = 1, batch_size do
        input_max_len = math.max(input_max_len, input[s+i-1]:nElement()/input_feature_dim)
        output_max_len = math.max(output_max_len, output[s+i-1]:nElement())
    end
    local batch_input = torch.Tensor(batch_size, input_max_len, input_feature_dim):fill(params.start_symbol)
    local batch_output = torch.Tensor(batch_size, output_max_len):fill(y_vocab)
    local xlength = torch.Tensor(batch_size):zero()
    local ylength = torch.Tensor(batch_size):zero()
    for i = 1, batch_size do
        local input_sequence_length = input[s+i-1]:nElement() / input_feature_dim
        batch_input[{i, {1, input_sequence_length}, {}}]:copy(input[s+i-1]:reshape(input_sequence_length, input_feature_dim))
        batch_output[{i, {1,output[s+i-1]:nElement()}}]:copy(output[s+i-1])
        if params.input_temporalconv_stride > 0 then
            xlength[i] = math.floor((input_sequence_length - params.input_temporalconv_width)/params.input_temporalconv_stride) + 1 -- Using temporal convolution
        else
            xlength[i] = input_sequence_length
        end
        ylength[i] = output[s+i-1]:nElement()
    end
--    local debugger = require('fb.debugger')
--    debugger.enter()

    if params.temporal_sampling == 'TemporalSampling' then
        for i = 1, batch_size do
            xlength[i] = math.floor(xlength[i] / params.temporalsampling_stride)
            if xlength[i] == 0 then
                xlength[i] = 1
            end
        end
    elseif params.temporal_sampling == 'TemporalConvolution' then
        for i = 1, batch_size do
            xlength[i] = math.floor((xlength[i] - params.temporalconv_width) / params.temporalconv_stride) + 1
        end
    end

    return batch_input, batch_output, xlength, ylength
end

local log0 = -1000000.
function prepare_minibatch(input, output, s, t, x_vocab, y_vocab, end_symbol, conv_dW_size, params)
  local input_max_len = 0
  local output_max_len = 0
  local batch_size = t - s + 1
  for i = 1, batch_size do
    input_max_len = math.max(input_max_len, input[s+i-1]:nElement()) 
    output_max_len = math.max(output_max_len, output[s+i-1]:nElement()) 
  end
  if end_symbol and end_symbol > 0 then
    input_max_len = input_max_len + 1
    output_max_len = output_max_len + 1
  end
  local batch_input = torch.Tensor(batch_size, input_max_len):fill(params.start_symbol)
  local mask_input = torch.Tensor(batch_size, input_max_len):fill(log0) -- logspace
  local batch_output = torch.Tensor(batch_size, output_max_len):fill(y_vocab)
  local xlength = torch.Tensor(batch_size):zero()
  local ylength = torch.Tensor(batch_size):zero()
  for i = 1, batch_size do
    batch_input[{i, {1,input[s+i-1]:nElement()}}]:copy(input[s+i-1])
    mask_input[{i, {1, input[s+i-1]:nElement()}}]:zero() -- logspace
    batch_output[{i, {1,output[s+i-1]:nElement()}}]:copy(output[s+i-1])
    xlength[i] = input[s+i-1]:nElement()
    ylength[i] = output[s+i-1]:nElement()
    if end_symbol and end_symbol > 0 then
      batch_input[{i, input[s+i-1]:nElement()+1}] = end_symbol
      mask_input[{i, input[s+i-1]:nElement()+1}] = 0
      batch_output[{i, output[s+i-1]:nElement()+1}] = end_symbol
      xlength[i] = xlength[i] + 1
      ylength[i] = ylength[i] + 1
    end
  end

    if params.input_temporalconv_stride > 0 then
    input_max_len  = math.floor((input_max_len - 1)/params.input_temporalconv_stride) + 1 -- Using temporal convolution
    mask_input:resize(batch_size, input_max_len):fill(log0)
    for i = 1, batch_size do
      xlength[i] = math.floor((xlength[i] - 1)/params.input_temporalconv_stride) + 1 -- Using temporal convolution
      mask_input[{i, {1, xlength[i]}}] = 0
    end
  end

  if conv_dW_size then
    input_max_len = math.floor((input_max_len - 1) / conv_dW_size) + 1
    mask_input:resize(batch_size, input_max_len):fill(log0)
    for i = 1, batch_size do
      xlength[i] = math.floor((xlength[i] - 1) / conv_dW_size) + 1
      mask_input[{i, {1, xlength[i]}}] = 0
    end
  end

  return batch_input, batch_output, xlength, ylength, mask_input
end

function g_cloneManyTimes(net, T)
  net:clearState()
  local clones = {}
  local params, gradParams = net:parameters()
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end

function toy_evaluate_predict_loss_mapping(ypred, yref, use_crossentropy, mapping_path, without_mapping)
    local phones_mapping = {}
    if without_mapping then
        for _id = 1, 29 do
            phones_mapping[_id] = _id
        end
    else
        for line in io.lines('%s/phones.60-39.mapping' % mapping_path) do
            splits = line:split('\t')
            phones_mapping[tonumber(splits[1]) + 1] = splits[2]
        end
    end

    assert(#ypred == #yref)

    local count = 0
    local loss = 0
    local WER = 0
    local total_num_words = 0

    local sequenceError = SequenceError()
    for i = 1, #ypred do
        local ypred_mapping = ''
        local prevalue = -1
        for j = 1, ypred[i]:nElement() do
            if use_crossentropy then
                _value = torch.floor((ypred[i][j] - 1) / 3) + 1
            else
                _value = ypred[i][j]
            end
            if phones_mapping[_value] then
                ypred_mapping = ypred_mapping .. ' ' .. phones_mapping[_value]
            end
        end

        local yref_mapping = ''
        local prevalue = -1
        for j = 1, yref[i]:nElement() do
            if use_crossentropy then
                _value = torch.floor((yref[i][j] - 1) / 3) + 1
            else
                _value = yref[i][j]
            end
            if phones_mapping[_value] then -- and _value ~= prevalue then
                yref_mapping = yref_mapping .. ' ' .. phones_mapping[_value]
            end
        end

        local word_errs , num_words = sequenceError:calculateWER(yref_mapping:gsub("^%s*(.-)%s*$", "%1"), ypred_mapping:gsub("^%s*(.-)%s*$", "%1"))
        WER = WER + word_errs
        total_num_words = total_num_words + num_words

        -- unmerged edit distance, count
        loss = loss + EditDistance(torch.totable(ypred[i]:long()), torch.totable(yref[i]:long()))
        count = count + yref[i]:nElement()
    end
    print('WER: ', WER, ' total num words: ', total_num_words)
    print(' ==> test predict WER: ', WER / total_num_words * 100)
    print(' ==> test predict edit distance (normalized): ', loss / count)
    return WER / total_num_words * 100
end


function toy_evaluate_predict_loss(ypred, yref)
    assert(#ypred == #yref)

    local count = 0
    local loss = 0
    for i = 1, #ypred do
        loss = loss + EditDistance(torch.totable(ypred[i]:long()), torch.totable(yref[i]:long()))
        count = count + yref[i]:nElement()
    end
    return loss / count
end

function generate_batches_by_length(input_train, batch_size, req_same_length)
  --- input train must have sorted
  local batches = {}
  if req_same_length then
    local count = 0
    local cur_len = input_train[1]:nElement()
    local s = 1
    local t = 1
    for i = 1, #input_train do
      if count < batch_size and input_train[i]:nElement() == cur_len then
        t, count = i, count + 1
      else 
        table.insert(batches, {s, t})
        s, t = i, i
        count = 1
        cur_len = input_train[i]:nElement()
      end
    end 
    if count > 0 then
      table.insert(batches, {s, t})
    end
  else
    local num_batches = math.ceil(#input_train / batch_size)
    for i = 1, num_batches do
      local s = (i-1)*batch_size + 1
      local t = math.min(s + batch_size - 1, #input_train)
      table.insert(batches, {s,t})
    end
  end
  return batches
end

---- machine translation specific ---

function load_data(filename, note, type)
  local note = note or filename
  print(string.format("--loading %s data from %s", note, filename))
  local data = {}
  for line in io.lines(filename) do
    splits = line:split(" ")
    if type == 'Double' then
        table.insert(data, torch.DoubleTensor(splits))
    else
        table.insert(data, torch.LongTensor(splits))
    end
  end
  print(string.format("  %s data size: %d", note, #data))
  return data
end

function load_vocab(filename, note)
  local note = note or filename
  print(string.format("--loading %s vocab from %s", note, filename))
  local vocab = {}
  for line in io.lines(filename) do
    splits = line:split(" ")
    if splits[1] == '' then
      vocab[tonumber(splits[2])] = ' '
    else
      vocab[tonumber(splits[2])] = splits[1]
    end
    if (not splits[2]) or splits[3] then
      print("error")
    end
  end
  -- the actual size is #vocab + 1, since we use 0 as padding
  print(string.format("  %s vocab size: %d", note, #vocab))
  return vocab
end

function tensor_to_string(x)
  local phrases = {}
  for i = 1, x:nElement() do
    table.insert(phrases, x[i])
  end
  return table.concat(phrases, ' ')
end

function decipher_tables(x, vocab)
    local decipher_tables = {}
    -- sed -r 's/(@@ )|(@@ ?$)//g'
    for i = 1, #x do
        local decipher_str = decipher(x[i]:view(-1, 1), vocab)
        local decipher_str_merge = string.gsub(string.gsub(decipher_str, "@@  ", ""), "@@ ", "")
        table.insert(decipher_tables, decipher_str_merge)
    end
    return decipher_tables
end


function decipher(x, ivocab)
  local phrases = {}
  for i = 1, x:nElement() do
    local phrase = ivocab[torch.totable(x[i])[1]] or 'UNK'
    table.insert(phrases, phrase)
  end
  return table.concat(phrases, ' ')
end

function eval_bleu_score(refs, outputs, work_dir, iter, unk_as_error, config_dir, unk_symbol, is_string)
  assert(#outputs == #refs)
  local ref_filename
  if is_string then
      ref_filename = string.format("%s/refs-at-iter-%d.txt", work_dir, iter)
      ref_f = assert(io.open(ref_filename, "w"))
      for i = 1, #refs do
          ref_f:write(refs[i], "\n")
      end
      ref_f:close()
  else  -- otherwise, table of tensor
      if unk_as_error then
        ref_filename = string.format("%s/refs-at-iter-%d-unk-as-error.txt", work_dir, iter)
        ref_f = assert(io.open(ref_filename, "w"))
        for i = 1, #refs do
          local new_ref = refs[i]:long() - torch.cmul(refs[i]:long(), refs[i]:eq(unk_symbol):long())
          ref_f:write(table.concat(new_ref:totable(), " "), "\n")
        end
        ref_f:close()
      else
        ref_filename = string.format("%s/refs-at-iter-%d.txt", work_dir, iter)
        ref_f = assert(io.open(ref_filename, "w"))
        for i = 1, #refs do
          ref_f:write(table.concat(refs[i]:totable(), " "), "\n")
        end
        ref_f:close()
      end
  end

  local output_filename
  if unk_as_error then
    output_filename = string.format("%s/results-at-iter-%d-unk-as-error.txt", work_dir, iter)
  else
    output_filename = string.format("%s/results-at-iter-%d.txt", work_dir, iter)
  end
  local out_f = assert(io.open(output_filename, "w"))

  if is_string then
      for i = 1, #outputs do
        out_f:write(outputs[i], "\n")
      end
  else  -- otherwise, table of tensor
      for i = 1, #outputs do
        out_f:write(table.concat(outputs[i]:totable(), " "), "\n")
      end
  end
  out_f:close()
  if config_dir == nil then
      config_dir = './'
  end
  local cmd = string.format("perl %s/data_processing/multi-bleu.perl %s < %s", config_dir, ref_filename, output_filename)
  os.execute(cmd)
  local cmd = string.format("perl %s/data_processing/multi-bleu.perl %s < %s", config_dir, ref_filename, output_filename)
  local handle = io.popen(cmd)
  local result = handle:read("*a")
  handle:close()
  _, str_end = string.find(result, 'EVALERR =')
  return tonumber(string.sub(result, str_end + 1))
end

function eval_wer_score(refs, outputs, work_dir, iter, without_mapping)
    assert(#outputs == #refs)
    local ref_filename

    ref_filename = string.format("%s/refs-at-iter-%d.txt", work_dir, iter)
    ref_f = assert(io.open(ref_filename, "w"))
    for i = 1, #refs do
        ref_f:write(i, ' ', table.concat(refs[i]:totable(), " "), "\n")
    end
    ref_f:close()
    --    end

    local output_filename
    output_filename = string.format("%s/results-at-iter-%d.txt", work_dir, iter)
    local out_f = assert(io.open(output_filename, "w"))
    for i = 1, #outputs do
        out_f:write(i, ' ', table.concat(outputs[i]:totable(), " "), "\n")
    end
    out_f:close()
    if without_mapping then
        local cmd = string.format("sh /home/pshuang/work/tools/kaldi/egs/timit/s5/local/score_timit_womapping.sh %s %s %s %s", output_filename, ref_filename, work_dir, iter)
        os.execute(cmd)
    else
        local cmd = string.format("sh /home/pshuang/work/tools/kaldi/egs/timit/s5/local/score_timit.sh %s %s %s %s", output_filename, ref_filename, work_dir, iter)
        os.execute(cmd)
    end
end

function yref_totable(yref, ylength)
  local bsize = yref:size(1)
  local yref_tab = {}
  for i = 1, bsize do
    table.insert(yref_tab, torch.totable(yref[{i, {1, ylength[i]}}]))
  end
  return yref_tab
end

function ctc_decodeOutput(predictions)
    --[[
        Turns the predictions tensor into a list of the most likely tokens
        NOTE:
            to compute WER we strip the begining and ending spaces
    --]]
    local tokens = {}
    local blankToken = 0
    local preToken = blankToken
    -- The prediction is a sequence of likelihood vectors
    local _, maxIndices = torch.max(predictions, 2)
    maxIndices = maxIndices:float():squeeze()

    for i=1, maxIndices:size(1) do
        local token = maxIndices[i] - 1 -- CTC indexes start from 1, while token starts from 0
        -- add token if it's not blank, and is not the same as pre_token
        if token ~= blankToken and token ~= preToken then
            table.insert(tokens, token)
        end
        preToken = token
    end
    return torch.Tensor(tokens)
end

function ctc_beam_search(predictions, configs)
    local beam_size = configs.beam_size or 40
    local use_avg_prob = configs.use_avg_prob
    local softmax = nn.LogSoftMax():cuda()
    local preds = softmax:forward(predictions):double()

    local T, V = preds:size(1), preds:size(2)
    local B = {{}}
    local B_inv = nil
    local Pr = {0}
    local pnb_ = {-torch.loginf()}
    local pb_ = {0}
    for t = 1, T do
      local B_new, Pr_new, pnb_new, pb_new  = {}, {}, {}, {}
      local _, ind = torch.sort(torch.Tensor(Pr), true)
      B_inv = {}
      for i = 1, math.min(beam_size, ind:nElement()) do
        local j = ind[i]
        B_inv[table.concat(B[j], "-")] = j
      end
      for i = 1, math.min(beam_size, ind:nElement()) do
        local j = ind[i]
        local y = B[j]
        local pnb = -torch.loginf()
        if #y > 0 then
          pnb = pnb_[j] + preds[{t, y[#y]+1}]
          local y_1_str = ''
          if #y > 1 then
            y_1_str = table.concat(torch.totable(torch.Tensor(y)[{{1,#y-1}}]), "-")
          end
          local jj = B_inv[y_1_str]
          if jj ~= nil then
            local y_1 = B[jj]
            if y_1_str:len() > 0 and y[#y] == y_1[#y_1] then
              pnb = torch.logadd(pnb, pb_[jj] + preds[{t, y[#y]+1}]) 
            else
              pnb = torch.logadd(pnb, Pr[jj] + preds[{t, y[#y]+1}])
            end
          end
        end
        local pb = Pr[j] + preds[{t, 1}] -- 1 is the blank symbol
        table.insert(B_new, torch.copy_array(y))
        table.insert(Pr_new, torch.logadd(pnb, pb))
        table.insert(pnb_new, pnb)
        table.insert(pb_new, pb)
        for v = 2, V do
          pb = -torch.loginf()
          if #y > 0 and v-1 == y[#y] then
            pnb = pb_[j] + preds[{t, v}]
          else
            pnb = Pr[j] + preds[{t, v}]
          end
          table.insert(pb_new, pb)
          table.insert(pnb_new, pnb)
          table.insert(Pr_new, torch.logadd(pnb, pb))
          local y_ = torch.copy_array(y)
          table.insert(y_, v-1)
          table.insert(B_new, y_) 
        end
      end
      B = B_new
      Pr = Pr_new
      pnb_ = pnb_new
      pb_ = pb_new
    end
    if use_avg_prob then
      for i = 1, #Pr do
        Pr[i] = Pr[i] / #(B[i])
      end
    end
    local _, indx = torch.sort(torch.Tensor(Pr), true)
    return torch.Tensor(B[indx[1]])
end

function xent_decodeOutput(predictions)
    --[[
        Turns the predictions tensor into a list of the most likely tokens
        NOTE:
            to compute WER we strip the begining and ending spaces
    --]]
    local tokens = {}
    local _, maxIndices = torch.max(predictions, 2)
    maxIndices = maxIndices:float():squeeze()
    for i=1, maxIndices:size(1) do
        local token = maxIndices[i]
        table.insert(tokens, token)
    end
    return torch.Tensor(tokens)
end

function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

function line_from(file)
  if not file_exists(file) then return 0 end
  local f = io.open(file, "r")
  io.input(f)
  local data = io.read()
  io.close(f)
  return data
end

function lines_from(file, type)
    if not file_exists(file) then return 0 end
    local data = {}
    for line in io.lines(file) do
        line = string.gsub(line, "\n", "")
        if type == 'Int' then
            table.insert(data, tonumber(line))
        elseif type == 'phrases' then
            local line_split = line:split(" ")
            local subtable = {}
            for i = 1, #line_split do
                table.insert(subtable, tonumber(line_split[i]))
            end
            table.insert(data, table.concat(subtable, '-'))
        else
            table.insert(data, line)
        end
    end
    return data
end


function Set(list)
    local set = {}
    for _, l in ipairs(list) do set[l] = true end
    return set
end

function tensor_to_table(input)
    local table_input = input:totable()
    return Set(table_input)
end

function table.slice(tbl, first, last, step)
    local sliced = {}
    for i = first or 1, last or #tbl, step or 1 do
        sliced[#sliced+1] = tbl[i]
    end
    return sliced
end
