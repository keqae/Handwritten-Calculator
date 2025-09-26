
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
    
    return eval(expression)


symbols = [7, 12, 9]

print(f"7*9 = {parse(symbols)}")