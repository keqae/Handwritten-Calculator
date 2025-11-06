
labels = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "+",
    11: "-",
    12: "*",
    13: "/",
}


def parse(classified: list[int]) -> str:
    expression = ""

    # iterate through classified symbol list and form the expression
    for symbol_id in classified:
        expression += labels[symbol_id]

    print(expression)

    try: 
        result = eval(expression)

        result_dict = {
            "expression": expression,
            "result": result
            }
        
    except ZeroDivisionError:
        result_dict = {
            "expression": expression,
            "result": "MATH ERROR: Can't divide by 0"
            }
    
    except SyntaxError:
        result_dict = {
            "expression": expression,
            "result": "SYNTAX ERROR"
            }
        
    return result_dict

