# encoding: utf-8
import codecs
import sys
import munkres
from wagnerfischerpp import WagnerFischer
import numpy
import re


def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


def load_lexicon(f):
  lexicon = dict()
  for line in f:
    word_and_pron = line.split(None, 1)
    if (len(word_and_pron) == 0):
      print "Invalid line: ", line.encode("utf-8");
      sys.exit(1)
    elif(len(word_and_pron) == 1):
      word = word_and_pron[0]
      pron = ""
    else:
      word = word_and_pron[0]
      pron = word_and_pron[1]
      pron = re.sub(r'_.', "", pron)
      pron = re.sub(r'\s+', " ", pron)
      pron = re.sub(r'\s+$', "", pron)

      try:
        maybe_cost, pron_aux = pron.split(None, 1)
        if (is_number(maybe_cost)):
          pron = pron_aux
      except ValueError:
        pass
    
    prons = lexicon.get(word, list())
    if pron in prons:
      pass
    else:
      prons.append(pron)
    lexicon[word] = prons

  return lexicon


def evaluate_g2p(ref, out):
  words1 = set(ref.keys())
  words2 = set(out.keys())
  pp = words1.union(words2)
  if len(pp) != len(words1):
    print("Error: test dictionary contains words that are not in the ref dict")
    print(pp.difference(words1))

  nof_phones = 0.0
  nof_phones_with_skipped = 0.0
  nof_phone_errs = 0.0
  nof_phone_errs_with_skipped = 0.0
  nof_word_errs = 0.0
  nof_phone_ins = 0.0
  nof_phone_del = 0.0
  nof_phone_sub = 0.0

  # words1 = [u'öf']
  for word in sorted(list(words1)):
    assert(ref.get(word, None) is not None)
    generated = list(out.get(word, list()))
    errs, phones, info = score_word_prons_SEQUITUR(word, list(ref[word]), generated)
    nof_phone_errs_with_skipped += errs
    nof_phones_with_skipped += phones
    if generated:
      nof_phone_errs += errs
      nof_phones += phones
      nof_phone_ins += info[0]
      nof_phone_del += info[1]
      nof_phone_sub += info[2]
    if errs > 0:
      nof_word_errs += 1.0


  l1 = float(len(words1))
  l2 = float(len(words2))
  r1 = 100 *l2/l1
  r2 = 100 * float(nof_phones)/nof_phones_with_skipped
  print "\ntotal: %d strings, %d symbols" % (len(words1), nof_phones_with_skipped)
  print "successfully translated: %d (%0.2f%%) strings, %d (%0.2f%%) symbols" % (l2, r1, nof_phones, r2)
  print "\tstring errors:  %d (%0.2f%%)" % (nof_word_errs, 100*float(nof_word_errs)/l2)
  print "\tsymbol errors:  %d (%0.2f%%)" % (nof_phone_errs, 100*float(nof_phone_errs)/nof_phones)
  print "\t\tinsertion errors   :  %d (%0.2f%%)" % (nof_phone_ins, 100*float(nof_phone_ins)/nof_phones)
  print "\t\tdeletion errors    :  %d (%0.2f%%)" % (nof_phone_del, 100*float(nof_phone_del)/nof_phones)
  print "\t\tsubstitution errors:  %d (%0.2f%%)" % (nof_phone_sub, 100*float(nof_phone_sub)/nof_phones)

  d = nof_phones_with_skipped - nof_phones
  r = float(d) / nof_phones_with_skipped
  w = nof_word_errs + (l1 - l2)
  print "translation failed: %d (%0.2f%%) strings, %d (%0.2f%%) symbols" % (l1 - l2, 100.0 * (l1 -l2)/float(l1), d, 100 * r)
  print "\tstring errors:  %d (%0.2f%%)" % (w, 100*float(w)/l1)
  print "\tsymbol errors:  %d (%0.2f%%)" % (nof_phone_errs_with_skipped, 100*float(nof_phone_errs_with_skipped)/nof_phones_with_skipped)

  #print "PER: ", nof_phone_errs / nof_phones
  #print "PER(including skipped): ", \
  #      nof_phone_errs_with_skipped / nof_phones_with_skipped
  #print "WER: ", nof_word_errs / len(words2)
  #print "WER(including skipped): ", nof_word_errs / len(words1)

  recall_sum = 0.0
  precision_sum = 0.0
  for word in words1:
    assert(ref.get(word, None) is not None)
    generated = list(out.get(word, list()))
    precision, recall = score_word_prons_IR(word, list(ref[word]), generated)
    precision_sum += precision
    recall_sum += recall

  precision = precision_sum / len(words1)
  recall = recall_sum / len(words1)
  f1 = 2 * (precision * recall) / (precision + recall)
  print ""
  print "PRECISION: %0.2f%%" % (100 * precision)
  print "RECALL   : %0.2f%%" % (100 * recall)
  print "F1       : %0.2f%%" % (100 * f1)
  print ""


def create_cost_matrix(word, ref, out):
  l1 = len(ref)
  l2 = len(out)
  if l2 <= 0:
    return None
  costs = numpy.zeros((l1, l2,))
  for i, pron1 in enumerate(ref):
    for j, pron2 in enumerate(out):
      edit = WagnerFischer(pron1.split(), pron2.split())
      costs[i][j] = edit.cost
  return costs


def compute_optimal_pairs(word, ref, out, costs, alpha=0.0):
  l1 = len(ref)
  l2 = len(out)
  if l2 <= 0:
    return set(), set([i for i in range(l1)]), set()
  m = munkres.Munkres()
  indexes = m.compute(costs.tolist())

  total = 0
  rows = set(range(l1))
  columns = set(range(l2))

  matches = set()
  for row, column in indexes:
    value = costs[row][column]
    if (value <= alpha):
      matches.add((row, column,))
      rows.remove(row)
      columns.remove(column)

  return matches, rows, columns


def score_word_prons_SEQUITUR(word, ref, out):
  if len(out) > 0:
    costs = create_cost_matrix(word, ref, [out[0]])
  else:
    costs = create_cost_matrix(word, ref, out)
  matches, nf, fa = compute_optimal_pairs(word, ref, out, costs, alpha=99999)
  assert(len(matches) <= 1)

  for i, j in list(matches):
    edit = WagnerFischer(ref[i].split(), out[j].split())
    # print word, edit.cost,  len(ref[i].split())
    ali = edit.alignments().next()
    #print ref[i]
    #print out[j]
    #print ali
    ins = ali.count('I')
    dele = ali.count('D')
    sub = ali.count('S')
    return edit.cost, len(ref[i].split()), (ins, dele, sub,)
  # return 0.0, 0.0
  return len(ref[0].split()), len(ref[0].split()), (len(ref[0].split()), 0, 0,)


def score_word_prons_SEQUITUR_2(word, ref, out):
  max_cost = 99999
  costs = create_cost_matrix(word, ref, out)
  matches, nf, fa = compute_optimal_pairs(word, ref, out, costs, alpha=max_cost)
  
  min_cost = max_cost
  min_i = 0
  min_j = 0
  for i, j in list(matches):
    edit = WagnerFischer(ref[i].split(), out[j].split())
    if edit.cost < min_cost:
      min_cost = edit.cost
      min_i = i
      min_j = j
    if min_cost == 0:
      break

  if min_cost < max_cost:
    #print "XX: %s, min_cost = %d" % (word, 0)
    edit = WagnerFischer(ref[min_i].split(), out[min_j].split())
    print "%s %s (%d errors)" % (word, out[min_j], edit.cost)
    return edit.cost, len(ref[min_i].split())
  # return 0.0, 0.0
  return len(ref[0].split()), len(ref[0].split())


def score_word_prons_IR(word, ref, out):
  costs = create_cost_matrix(word, ref, out)
  matches, nf, fa = compute_optimal_pairs(word, ref, out, costs, alpha=1.1)

  precision = 0
  if len(out) > 0:
    precision = len(matches) / len(out)
  recall = len(matches) / len(ref)
  # print word
  for i, j in list(matches):
    edit = WagnerFischer(ref[i].split(), out[j].split())
    # print "MATCH:[", edit.cost, "]", ref[i], " -> ", out[j]
    pass
  for i in list(nf):
    # print "MISS: ", ref[i]
    pass
  for j in list(fa):
    # print "FA:   ", out[j]
    pass

  return precision, recall


if __name__ == "__main__":
  with codecs.open(sys.argv[1], 'rb', 'utf-8') as f:
    ref = load_lexicon(f)

  print("From ref file %s loaded %d words" % (sys.argv[1], len(ref),))

  with codecs.open(sys.argv[2], 'rb', 'utf-8') as f:
    out = load_lexicon(f)

  print("From test file %s loaded %d words" % (sys.argv[2], len(out),))

  out = evaluate_g2p(ref, out)
