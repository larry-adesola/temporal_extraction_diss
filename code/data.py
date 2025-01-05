from enum import Enum
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET

POS_VOCAB = {}
def get_pos_id(pos):
    
    if pos not in POS_VOCAB:
        POS_VOCAB[pos] = len(POS_VOCAB)  # assign next available ID
    return POS_VOCAB[pos]

class LabelType(Enum):
    BEFORE = 0
    AFTER = 1
    EQUAL = 2
    VAGUE = 3

    @staticmethod
    def to_class_index(label_type):
        for label in LabelType:
            if label_type == label.name:
                return label.value

class temprel_ee:
    def __init__(self, xml_element):
        self.xml_element = xml_element
        self.label = xml_element.attrib['LABEL']
        self.sentdiff = int(xml_element.attrib['SENTDIFF'])
        self.docid = xml_element.attrib['DOCID']
        self.source = xml_element.attrib['SOURCE']
        self.target = xml_element.attrib['TARGET']
        self.data = xml_element.text.strip().split()
        self.token = []
        self.lemma = []
        self.part_of_speech = []
        self.position = []
        self.length = len(self.data)
        self.event_ix = []
        self.text = ""
        self.event_offset = []

        is_start = True
        for i,d in enumerate(self.data):
            tmp = d.split('///')
            self.part_of_speech.append(tmp[-2])
            self.position.append(tmp[-1])
            self.token.append(tmp[0])
            self.lemma.append(tmp[1])

            if is_start:
                is_start = False
            else:
                self.text += " "

            if tmp[-1] == 'E1':
                self.event_ix.append(i)
                self.event_offset.append(len(self.text))
            elif tmp[-1] == 'E2':
                self.event_ix.append(i)
                self.event_offset.append(len(self.text))

            self.text += tmp[0]

        assert len(self.event_ix) == 2


class temprel_set:
  def __init__(self, xmlfname, datasetname="matres"):
      self.xmlfname = xmlfname
      self.datasetname = datasetname
      tree = ET.parse(xmlfname)
      root = tree.getroot()
      self.size = len(root)
      self.temprel_ee = []
      for e in root:
          self.temprel_ee.append(temprel_ee(e))

  def to_tensor(self, tokenizer, pos_enabled):

    gathered_text = [ee.text for ee in self.temprel_ee]
    tokenized_output = tokenizer(gathered_text, padding=True, return_offsets_mapping=True,)

    if pos_enabled:
      all_pos_ids = []
      
      for i, ee in enumerate(self.temprel_ee):
        offsets = tokenized_output['offset_mapping'][i]

        pos_ids_example = []
        
        # Step 1: Build a list of (start_char, end_char, pos_id) for each original token in ee
        #         so we can see which subword offset maps to which token.
        
        running_char = 0  
        token_spans = []
        for pos_tag, token_text in zip(ee.part_of_speech, ee.token):
          pos_tag_id = get_pos_id(pos_tag)  
          start_char = running_char
          end_char = running_char + len(token_text)
          token_spans.append((start_char, end_char, pos_tag_id))
          running_char += len(token_text) + 1  
        
        # Step 2: For each subword offset, find which original token span it belongs to
        for (sub_start, sub_end) in offsets:
          assigned_pos = 0  # default if we fail to match, or handle [CLS],[SEP] etc.
          
          # We skip special tokens with offset (0,0), or negative
          if (sub_start == 0 and sub_end == 0):
            pos_ids_example.append(assigned_pos)
            continue
          
          # Find which token span covers sub_start..sub_end
          for (tok_start, tok_end, pos_id) in token_spans:
            # If the subword start is within this token, we assign that pos_id
            # Some do an overlap check (sub_start < tok_end and sub_end > tok_start)
            if tok_start <= sub_start < tok_end:
              assigned_pos = pos_id
              break
          pos_ids_example.append(assigned_pos)
        
        all_pos_ids.append(pos_ids_example)
      
      pos_ids = torch.LongTensor(all_pos_ids)




    tokenized_event_ix = []

    for i in range(len(self.temprel_ee)):

      event_ix_pair = []
      for j, offset_pair in enumerate(tokenized_output['offset_mapping'][i]):
          if (offset_pair[0] == self.temprel_ee[i].event_offset[0] or\
              offset_pair[0] == self.temprel_ee[i].event_offset[1]) and\
              offset_pair[0] != offset_pair[1]:
              event_ix_pair.append(j)
      if len(event_ix_pair) != 2:
          raise ValueError(f'Instance {i} doesn\'t found 2 event idx.')
      tokenized_event_ix.append(event_ix_pair)
    input_ids = torch.LongTensor(tokenized_output['input_ids'])
    attention_mask = torch.LongTensor(tokenized_output['attention_mask'])
    tokenized_event_ix = torch.LongTensor(tokenized_event_ix)
    labels = torch.LongTensor([LabelType.to_class_index(ee.label) for ee in self.temprel_ee])

    if pos_enabled:
      return TensorDataset(input_ids, attention_mask, tokenized_event_ix, labels, pos_ids)
    return TensorDataset(input_ids, attention_mask, tokenized_event_ix, labels)




