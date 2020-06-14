import plac
import json
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

def prepare_train_data(input_filepath):
    resp = []
    try:
        with open(input_filepath, 'r', encoding='utf-8-sig') as rf:
            raw_json = json.load(rf)
        raw_data = raw_json.get("rasa_nlu_data")
        if not raw_data: return []
        raw_data = raw_data.get("common_examples")
        if not raw_data: return []
        for datum in raw_data:
            text = datum.get('text')
            entities = []
            for entity in datum.get('entities'):
                entities.append((entity.get('start'), entity.get('end'), entity.get('entity')))
            resp.append((text, {"entities": entities}))
        return resp
    except FileNotFoundError:
        print(f"file not found {input_filepath}, check you filepath")
        return []
    except json.JSONDecodeError:
        print(f"invalid json file")
        return []
    except Exception as err:
        print(err)
        return []
 
# Change training_file.json to your .json training file
# Change output_dir to path where we want to store new module    
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def train(input_filepath = "training_file.json", output_dir = "Path_where_it_store_module", model=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    TRAIN_DATA = prepare_train_data(input_filepath)
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for _ in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses
                )
            print("Losses", losses)

    # # test the trained model
    # for text, _ in TRAIN_DATA:
    #     doc = nlp(text)
    #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    #     print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # # test the saved model
        # print("Loading from", output_dir)
        # nlp2 = spacy.load(output_dir)
        # for text, _ in TRAIN_DATA:
        #     doc = nlp2(text)
        #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        #     print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

if __name__ == "__main__":
    plac.call(train)
    

# Ongoing
import spacy
nlp2 = spacy.load("Path/")

data = "Testing sentence"
data = nlp2(data)
for sent in data.ents:
    print("text : {} and label : {}".format(sent.text, sent.label_))





























