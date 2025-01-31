import os
import xml.etree.ElementTree as ET
import spacy


nlp = spacy.load("pt_core_news_sm")


def map_reltype_to_label(relType):
    """
    (BEFORE, AFTER, EQUAL, VAGUE).
    (e.g., treat OVERLAP as EQUAL or as VAGUE).
    """
    rel_upper = relType.upper()
    if rel_upper == "BEFORE":
        return "BEFORE"
    elif rel_upper == "AFTER":
        return "AFTER"
    elif rel_upper in ("SIMULTANEOUS", "IDENTITY", "OVERLAP", "OVERLAP-OR-AFTER"):
        return "EQUAL"  # or "VAGUE", depending on your preference
    else:
        return "VAGUE"


def tokenize_and_tag(sentence_text):
    
    doc = nlp(sentence_text)
    tokens = []
    for token in doc:
        # token.pos_ is a coarse POS tag in spaCy
        # token.lemma_ is the lemma
        # token.idx is start_char index
        start_char = token.idx
        end_char = start_char + len(token.text)
        tokens.append((token.text, token.lemma_, token.pos_, start_char, end_char))
    return tokens


def process_single_tml_file(input_file):
    """
    <SENTENCE DOCID="..." SOURCE="eX" TARGET="eY" ... LABEL="BEFORE">
      word1///lemma1///pos1///B word2///lemma2///pos2///E1 ...
    </SENTENCE>
    """
    # Parse the .tml
    tree = ET.parse(input_file)
    root = tree.getroot()  

    sentences = []
    s_nodes = list(root.findall(".//s"))  # all <s> in the file
    for i, s_node in enumerate(s_nodes):
        sent_id = i + 1
    
        s_text = ET.tostring(s_node, encoding="unicode", method="text").strip()

        token_list = tokenize_and_tag(s_text)
        sentences.append({
            "id": sent_id,
            "node": s_node,
            "text": s_text,
            "tokens": token_list
        })



    event_dict = {}


    all_event_elements = root.findall(".//EVENT")
    for ev_el in all_event_elements:
        eid = ev_el.attrib["eid"]  
        event_text = ev_el.text   

        s_idx = None
        offset_in_sentence = None

        # A naive approach
        for sent_obj in sentences:

            idx_found = sent_obj["text"].find(event_text)
            if idx_found != -1:
                s_idx = sent_obj["id"]
                offset_in_sentence = idx_found
                break

        # store info
        event_dict[eid] = {
            "element": ev_el,
            "event_text": event_text,
            "sentence_id": s_idx,
            "offset_in_sentence": offset_in_sentence
        }

    # Gather TLINKs that connect event->event
    event_links = []
    for tlink in root.findall("./TLINK"):
        if "eventID" in tlink.attrib and "relatedToEvent" in tlink.attrib:
            e1 = tlink.attrib["eventID"]
            e2 = tlink.attrib["relatedToEvent"]
            relType = tlink.attrib.get("relType", "VAGUE")
            label = map_reltype_to_label(relType)
            event_links.append((e1, e2, label))

    output_root = ET.Element("DATA")


    for (eid1, eid2, label) in event_links:
        if eid1 not in event_dict or eid2 not in event_dict:
          
          continue
        e1_info = event_dict[eid1]
        e2_info = event_dict[eid2]
        s_id1 = e1_info["sentence_id"]
        s_id2 = e2_info["sentence_id"]


        if s_id1 is None or s_id2 is None:
            continue
        if s_id1 != s_id2:
            #only handles same-sentence pairs, skip.
            continue

        # They are in the same sentence
        sent_id = s_id1

        sentence_obj = None
        for s_obj in sentences:
            if s_obj["id"] == sent_id:
                sentence_obj = s_obj
                break
        if sentence_obj is None:
            continue


        token_list = sentence_obj["tokens"]


        final_tokens = []
        e1_text = e1_info["event_text"]
        e2_text = e2_info["event_text"]


        found_e1 = False
        found_e2 = False

        for (tok, lemma, pos, start_char, end_char) in token_list:
            if not found_e1 and not found_e2:
              tag = "B"
            elif found_e1 and not found_e2:
              tag = "M"
            else:
              tag = "A"
            # naive check

            if not found_e1 and e1_text in tok:
                tag = "E1"
                found_e1 = True
            if not found_e2 and e2_text in tok:
                tag = "E2"
                found_e2 = True

            triple = f"{tok}///{lemma}///{pos}///{tag}"
            final_tokens.append(triple)

        # Build the <SENTENCE> element
        sent_el = ET.Element("SENTENCE")
        sent_el.set("DOCID", os.path.basename(input_file))
        sent_el.set("SOURCE", eid1)
        sent_el.set("TARGET", eid2)
        sent_el.set("SOURCE_SENTID", str(sent_id))
        sent_el.set("TARGET_SENTID", str(sent_id))
        sent_el.set("LABEL", label)
        sent_el.set("SENTDIFF", "0")  # same sentence

        # Join tokens with space or newline
        sent_el.text = " ".join(final_tokens)

        # Append to output
        output_root.append(sent_el)


    return output_root


def convert_timebankpt_dir(input_dir, output_xml):

    # Create a <DATA> root
    data_root = ET.Element("DATA")

    # Iterate .tml files
    for fname in os.listdir(input_dir):
        if not fname.endswith(".tml"):
            continue
        filepath = os.path.join(input_dir, fname)
        print(f"Processing file: {filepath}")
        sent_elems = process_single_tml_file(filepath)
        # Append these sentence elements to data_root
        for s_el in sent_elems:
            data_root.append(s_el)

    # Write to one XML file
    tree_out = ET.ElementTree(data_root)
    tree_out.write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"Done. Wrote all sentences to {output_xml}")





input_tml = "/content/drive/MyDrive/diss/train"
output_xml = "/content/drive/MyDrive/diss/train_pt.xml"

convert_timebankpt_dir(input_tml, output_xml)