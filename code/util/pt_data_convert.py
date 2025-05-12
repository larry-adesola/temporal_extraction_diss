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
    Processes a .tml file and produces <SENTENCE> elements.
    
    For same-sentence event pairs, the tokens come from a single sentence.
    For cross-sentence event pairs, tokens from the two sentences are processed
    separately and then concatenated (with a separator) in the output.
    
    The output element attributes include:
      DOCID, SOURCE, TARGET, SOURCE_SENTID, TARGET_SENTID, LABEL, SENTDIFF.
    
    Token tagging is done as follows:
      - For same-sentence events:
           Tokens overlapping event1 get tag "E1"
           Tokens overlapping event2 get tag "E2"
           Tokens before event1 get tag "B"
           Tokens between events get tag "M"
           All others get tag "A"
      - For cross-sentence events:
           In sentence 1 (with event1): tokens overlapping event1 get "E1",
           tokens before event1 get "B", and tokens after get "A".
           In sentence 2 (with event2): tokens overlapping event2 get "E2",
           tokens before event2 get "B", and tokens after get "A".
    """
    import os
    import xml.etree.ElementTree as ET

    # Parse the .tml file.
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Build a list of sentences.
    sentences = []
    s_nodes = list(root.findall(".//s"))  # all <s> elements
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

    # Build a dictionary of events.
    event_dict = {}
    all_event_elements = root.findall(".//EVENT")
    for ev_el in all_event_elements:
        eid = ev_el.attrib["eid"]
        event_text = ev_el.text

        s_idx = None
        offset_in_sentence = None

        # Use the XML structure so that we pick the sentence node that
        # actually contains this event element.
        for sent_obj in sentences:
            if sent_obj["node"].find(f".//EVENT[@eid='{eid}']") is not None:
                s_idx = sent_obj["id"]
                # Use .find() on the sentence text to get the offset.
                offset_in_sentence = sent_obj["text"].find(event_text)
                break

        # Store info (even if s_idx is None, we keep the event)
        event_dict[eid] = {
            "element": ev_el,
            "event_text": event_text,
            "sentence_id": s_idx,
            "offset_in_sentence": offset_in_sentence
        }

    # Gather TLINKs that connect event->event.
    event_links = []
    for tlink in root.findall("./TLINK"):
        if "eventID" in tlink.attrib and "relatedToEvent" in tlink.attrib:
            e1 = tlink.attrib["eventID"]
            e2 = tlink.attrib["relatedToEvent"]
            relType = tlink.attrib.get("relType", "VAGUE")
            label = map_reltype_to_label(relType)
            event_links.append((e1, e2, label))

    output_root = ET.Element("DATA")

    # Process each event link.
    for (eid1, eid2, label) in event_links:
        if eid1 not in event_dict or eid2 not in event_dict:
            continue

        e1_info = event_dict[eid1]
        e2_info = event_dict[eid2]
        s_id1 = e1_info["sentence_id"]
        s_id2 = e2_info["sentence_id"]

        # Skip if we cannot determine the sentence for either event.
        if s_id1 is None or s_id2 is None:
            continue

        # If both events are in the same sentence...
        if s_id1 == s_id2:
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
            e1_offset = e1_info["offset_in_sentence"]
            e1_end = e1_offset + len(e1_text) if e1_offset is not None else None
            e2_offset = e2_info["offset_in_sentence"]
            e2_end = e2_offset + len(e2_text) if e2_offset is not None else None

            for (tok, lemma, pos, start_char, end_char) in token_list:
                # Default tag.
                tag = "O"
                # Check if token overlaps event1.
                if e1_offset is not None and start_char <= e1_offset < end_char:
                    tag = "E1"
                # Check if token overlaps event2.
                elif e2_offset is not None and start_char <= e2_offset < end_char:
                    tag = "E2"
                else:
                    # Optionally assign tags based on positions.
                    if e1_offset is not None and start_char < e1_offset:
                        tag = "B"
                    elif (e1_offset is not None and e2_offset is not None and 
                          start_char > e1_offset and end_char < e2_offset):
                        tag = "M"
                    else:
                        tag = "A"
                triple = f"{tok}///{lemma}///{pos}///{tag}"
                final_tokens.append(triple)

            # Build the <SENTENCE> element.
            sent_el = ET.Element("SENTENCE")
            sent_el.set("DOCID", os.path.basename(input_file))
            sent_el.set("SOURCE", eid1)
            sent_el.set("TARGET", eid2)
            sent_el.set("SOURCE_SENTID", str(s_id1))
            sent_el.set("TARGET_SENTID", str(s_id2))
            sent_el.set("LABEL", label)
            sent_el.set("SENTDIFF", "0")  # same sentence
            sent_el.text = " ".join(final_tokens)
            output_root.append(sent_el)

        else:
            # Cross-sentence event pair.
            # Find the sentence objects for each event.
            sent_obj1 = None
            sent_obj2 = None
            for s_obj in sentences:
                if s_obj["id"] == s_id1:
                    sent_obj1 = s_obj
                if s_obj["id"] == s_id2:
                    sent_obj2 = s_obj
            if sent_obj1 is None or sent_obj2 is None:
                continue

            # Process tokens from the sentence containing event1.
            token_list1 = sent_obj1["tokens"]
            final_tokens1 = []
            e1_text = e1_info["event_text"]
            e1_offset = e1_info["offset_in_sentence"]
            for (tok, lemma, pos, start_char, end_char) in token_list1:
                tag = "O"
                if e1_offset is not None and start_char <= e1_offset < end_char:
                    tag = "E1"
                else:
                    if e1_offset is not None and start_char < e1_offset:
                        tag = "B"
                    else:
                        tag = "M"
                triple = f"{tok}///{lemma}///{pos}///{tag}"
                final_tokens1.append(triple)

            # Process tokens from the sentence containing event2.
            token_list2 = sent_obj2["tokens"]
            final_tokens2 = []
            e2_text = e2_info["event_text"]
            e2_offset = e2_info["offset_in_sentence"]
            for (tok, lemma, pos, start_char, end_char) in token_list2:
                tag = "O"
                if e2_offset is not None and start_char <= e2_offset < end_char:
                    tag = "E2"
                else:
                    if e2_offset is not None and start_char < e2_offset:
                        tag = "M"
                    else:
                        tag = "A"
                triple = f"{tok}///{lemma}///{pos}///{tag}"
                final_tokens2.append(triple)

            # Combine the token streams from the two sentences.
            # (use a " ||| " separator to indicate a sentence boundary;
            #  you can change or remove the separator as needed.)
            combined_tokens = " ".join(final_tokens1) + " " +" ".join(final_tokens2)

            # Build the <SENTENCE> element for cross-sentence events.
            sent_el = ET.Element("SENTENCE")
            sent_el.set("DOCID", os.path.basename(input_file))
            sent_el.set("SOURCE", eid1)
            sent_el.set("TARGET", eid2)
            sent_el.set("SOURCE_SENTID", str(s_id1))
            sent_el.set("TARGET_SENTID", str(s_id2))
            sent_el.set("LABEL", label)
            # Set SENTDIFF to the difference between sentence ids.
            sent_el.set("SENTDIFF", str(abs(s_id2 - s_id1)))
            sent_el.text = combined_tokens
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