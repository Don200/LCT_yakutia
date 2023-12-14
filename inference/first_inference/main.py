from fastapi import FastAPI
#from saigamodel import Conversation, Model


app = FastAPI(
    title='Kalhoznay zaglushka'
)
#conversation = Conversation()
#model = Model()

@app.get("/text")
def generate_offer(inputs: str):
    '''for inp in inputs:
        conversation.add_user_message(inp)
        prompt = conversation.get_prompt(model.tokenizer)
        output = model.generate(model, model.tokenizer, prompt, model.generation_config)

        return {"message": output}'''
    return inputs

'''
accelerate==0.21.0
bitsandbytes==0.40.2
peft==0.5.0
transformers==4.34.0
pandas~=2.1.4
'''

