from transformers import pipeline

class ExtractiveQnA_inference(config):
  def __init__(self, config):
    self.qa_pipeline = pipeline("question-answering",
                            model=config.DIRECTORY,
                            tokenizer=config.DIRECTORY)
  
  def predict(self, context, question):
    predictions = self.qa_pipeline({'context': context, 
                                    'question': question})
    
    return predictions