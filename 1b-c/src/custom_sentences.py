
def create_custom_sentence(input_dict):
    req_phrase = " and ".join([f"{req}" for req in input_dict.keys()])
    values = []
    
    for requirement, attributes in input_dict.items():
        for attribute, value in attributes.items():
            if attribute == "pricerange":
                description = f"the price is {value}"
            elif attribute == "foodquality":
                description = f"the food quality is {value}"
            elif attribute == "crowdedness":
                description = "it is not crowded" if value == "normal" else f"is it usually quite busy"
            elif attribute == "lengthofstay":
                description = f"you can stay for a {value} time"
            values.append(description)
    
    value_phrase = " and ".join(values)
    
    return f"system: This restaurant meets your {req_phrase} requirements, because {value_phrase}."