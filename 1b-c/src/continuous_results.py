def print_results(results):
    if len(results) > 1:
                print("system: So far, these are some of the restaurants that meet your preferences:\n")
                print(f"{'Name':<50} {'Food':<25} {'Price':<25} {'Area':<25}")
                print("-" * 125)
                count = 0
                for name, food, price, area in zip(results['restaurantname'], results['food'], results['pricerange'], results['area']):
                    if count < 10:
                        print(f"{name:<50} {food:<25} {price:<25} {area:<25}")
                        count += 1
                    else:
                        break
                print("-" * 125)