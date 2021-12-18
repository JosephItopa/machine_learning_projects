# import modules
from fastapi import FastAPI, Request
from app.simpleinterest import simple_interest

# instantiate fastapi
app= FastAPI()

@app.get("/")
async def get_input(request:Request):
    """
        Get inputs from users and call the simple interest function to evaluate
        the parameters then return the output. 
    """
    getInput= await request.json()
    principal= getInput['principal']
    rate= getInput['rate']
    period= getInput['period']

    # evaluate interest
    interest= simple_interest(principal, rate, period)

    return interest