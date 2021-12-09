def simple_interest(principal, rate, period):
    """
    A flask app to evaluate simple interest.
    """
    interest= (principal * rate * period)/100

    return interest