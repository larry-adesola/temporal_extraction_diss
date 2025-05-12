import spacy
import re
import pandas as pd
from typing import List, Tuple
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_utils')))

nlp = spacy.load("en_core_web_sm")

MODAL_WORDS = {"will", "would", "can", "could", "shall", "should", "may", "might", "must"}
SIMULT_WORDS = {"while", "simultaneously", "during", "meanwhile", "same", "time"}

def parse_sentence_spacy(sentence_text: str):
  """
  Parse the entire <SENTENCE> text with SpaCy for morphological and lexical cues.
  Returns the spaCy Doc object.
  """
  return nlp(sentence_text)

def find_event_tokens(doc, e1_text: str, e2_text: str) -> Tuple[int, int]:
  """
  Find the token indices in the spaCy Doc.
  """
  e1_idx, e2_idx = -1, -1
  e1_lower, e2_lower = e1_text.lower(), e2_text.lower()

  for i, token in enumerate(doc):
    tok_lower = token.text.lower()
    if tok_lower == e1_lower and e1_idx == -1:
      e1_idx = i
    elif tok_lower == e2_lower and e2_idx == -1:
      e2_idx = i

  return e1_idx, e2_idx

def approximate_tense_aspect_spacy(token):
  """
  For a single spaCy token,
  glean morphological Tense/Aspect. Returns (tense, aspect).
  """
  morph = token.morph
  tense = "OTHER"
  aspect = "NONE"

  if "Tense=Past" in morph:
    tense = "PAST"
  elif "Tense=Pres" in morph:
    tense = "PRESENT"
  elif "Tense=Fut" in morph:
    tense = "FUTURE"

  if "Aspect=Perf" in morph:
    aspect = "PERFECT"
  elif "Aspect=Prog" in morph or "Aspect=Imp" in morph:
    aspect = "PROGRESSIVE"

  return (tense, aspect)

def extract_features_full_sentence(ee):
  """
  For a temprel_ee object that has:
    - A single verb E1, a single verb E2
    - The entire sentence text in ee.text, or can be reconstructed from ee.data
    1) Parse with spaCy
    2) Locate E1 and E2 tokens
    3) Extract morphological (tense, aspect) for each event
    4) Check presence of modals / simult words in the entire sentence
    5) Gather distance or sentdiff
    6) Return the label (BEFORE, AFTER, EQUAL, VAGUE)
  """

  feats = {}

  sentence_text = ee.text  # assume ee.text is the entire sentence string

  # Parse with spaCy
  doc = parse_sentence_spacy(sentence_text)

  # identify single-verb strings for E1/E2
  if len(ee.event_ix) != 2:
    # fallback
    feats["tense_match"] = 0
    feats["aspect_match"] = 0
    feats["either_modal"] = 0
    feats["simult_word_present"] = 0
    feats["token_distance"] = 0
    feats["sentdiff"] = ee.sentdiff
    feats["label"] = ee.label
    return feats

  e1_ix, e2_ix = ee.event_ix

  # single verbs
  e1_verb_str = ee.token[e1_ix]  
  e2_verb_str = ee.token[e2_ix] 

  # Locate these tokens in the doc
  e1_doc_idx, e2_doc_idx = find_event_tokens(doc, e1_verb_str, e2_verb_str)

  # If found them, get morphological info
  if e1_doc_idx >= 0 and e1_doc_idx < len(doc):
    e1_tense, e1_aspect = approximate_tense_aspect_spacy(doc[e1_doc_idx])
  else:
    e1_tense, e1_aspect = ("OTHER", "NONE")

  if e2_doc_idx >= 0 and e2_doc_idx < len(doc):
    e2_tense, e2_aspect = approximate_tense_aspect_spacy(doc[e2_doc_idx])
  else:
    e2_tense, e2_aspect = ("OTHER", "NONE")

  feats["tense_match"] = 1 if e1_tense == e2_tense else 0
  feats["aspect_match"] = 1 if e1_aspect == e2_aspect else 0

  # Check presence of modals / simult words in entire sentence
  modal_found = False
  simult_found = False
  for token in doc:
      lower_tok = token.text.lower()
      if lower_tok in MODAL_WORDS:
          modal_found = True
      if lower_tok in SIMULT_WORDS:
          simult_found = True



  feats["either_modal"] = 1 if modal_found else 0
  feats["simult_word_present"] = 1 if simult_found else 0

  # Distance or sentdiff 
  feats["sentdiff"] = ee.sentdiff
  if e1_doc_idx >= 0 and e2_doc_idx >= 0:
      feats["token_distance"] = abs(e1_doc_idx - e2_doc_idx)
  else:
      feats["token_distance"] = 0

  #Label
  feats["label"] = ee.label  # BEFORE, AFTER, EQUAL, VAGUE

  return feats

def build_feature_dataframe_full_sentence(temprel_ee_list):
    rows = []
    for ee in temprel_ee_list:
      feats = extract_features_full_sentence(ee)
      rows.append(feats)
    return pd.DataFrame(rows)



def main():
  from data import temprel_set

  print("Logical Regression")
  train_set = temprel_set("../trainset-temprel.xml")
  train_ee_list = train_set.temprel_ee
  #
  # Then build the DataFrame:
  df_train = build_feature_dataframe_full_sentence(train_ee_list)
  #
  # At that point, you can do:
  X = df_train.drop("label", axis=1)
  y = df_train["label"]

  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import classification_report
  from imblearn.over_sampling import RandomOverSampler
  from sklearn.ensemble import RandomForestClassifier
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create the oversampler object
  ros = RandomOverSampler(random_state=42)

  # Fit the oversampler on your training data and resample
  X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

  # Now train your classifier using the resampled data
  clf = LogisticRegression(max_iter=200)
  clf.fit(X_resampled, y_resampled)

  # Evaluate on the original test set
  preds = clf.predict(X_test)
  print(classification_report(y_test, preds))
  print(clf.coef_)

  print("Random Forest")

  clf = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    random_state=42
  )
  clf.fit(X_resampled, y_resampled)

  # Evaluate on the original test set
  preds = clf.predict(X_test)
  print(classification_report(y_test, preds))

  #Examine feature importances
  importances = clf.feature_importances_
  feature_names = list(X.columns)
  for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
      print(f"{name}: {imp:.4f}")
    

if __name__ == "__main__":
    main()