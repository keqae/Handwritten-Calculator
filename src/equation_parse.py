
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
    for (symbol_id, idx) in classified:
        expression += labels[symbol_id]
    
    print(expression)
    
    return eval(expression)

