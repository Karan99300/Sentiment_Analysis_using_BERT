import torch
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr

label_dict={'neutral': 0,'negative': 1, 'positive': 2}

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)
model.load_state_dict(torch.load('finetuned_BERT_epoch_2.model',map_location='cpu'))
model.eval()

def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs.to('cpu')  
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return get_key_by_value(label_dict,predicted_class)

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    live=True,
    title="BERT Sentiment Analysis (CPU)",
    description="Enter a text and get sentiment prediction.",
)
iface.launch()
