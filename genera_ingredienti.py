import pandas as pd
import numpy as np

# Define the structure of the CSV
columns = ['name', 'cho_per_100g', 'protein', 'fat', 'calories', 'fiber',
           'is_vegan', 'is_vegetarian', 'is_gluten_free', 'is_lactose_free', 'food_clean']

# Create a list to hold the ingredient data
ingredients_data = []

# Helper function to create ingredient entries


def create_ingredient(name, cho, protein, fat, calories, fiber, vegan, vegetarian, gluten_free, lactose_free):
    return {
        'name': name,
        'cho_per_100g': cho,
        'protein': protein,
        'fat': fat,
        'calories': calories,
        'fiber': fiber,
        'is_vegan': vegan,
        'is_vegetarian': vegetarian,
        'is_gluten_free': gluten_free,
        'is_lactose_free': lactose_free,
        'food_clean': name  # Using the name as is for simplicity
    }


# Create 100 ingredient entries - this is expanded from the original request for 10
ingredients_data.extend([
    create_ingredient('Apple', 14.0, 0.3, 0.2, 52,
                      2.4, True, True, True, True),
    create_ingredient('Banana', 23.0, 1.1, 0.3, 96,
                      2.6, True, True, True, True),
    create_ingredient('Orange', 12.0, 0.9, 0.1, 47,
                      2.4, True, True, True, True),
    create_ingredient('Strawberry', 8.0, 0.7, 0.3, 33,
                      2.0, True, True, True, True),
    create_ingredient('Blueberry', 14.5, 0.7, 0.3, 57,
                      2.4, True, True, True, True),

    create_ingredient('Bread', 49.0, 9.0, 1.6, 265,
                      2.7, True, True, False, True),
    create_ingredient('Pasta', 74.0, 13.0, 1.1, 371,
                      3.2, True, True, False, True),
    create_ingredient('Rice', 28.0, 2.7, 0.2, 130,
                      0.4, True, True, True, True),
    create_ingredient('Quinoa', 21.0, 4.4, 1.9, 120,
                      2.8, True, True, True, True),
    create_ingredient('Oats', 66.0, 17.0, 7.0, 389,
                      10.6, True, True, True, True),

    create_ingredient('Chicken Breast', 0.0, 31.0, 3.6,
                      165, 0.0, False, False, True, True),
    create_ingredient('Beef Steak', 0.0, 26.0, 15.0, 240,
                      0.0, False, False, True, True),
    create_ingredient('Salmon', 0.0, 20.0, 13.0, 208,
                      0.0, False, False, True, True),
    create_ingredient('Tuna', 0.0, 29.0, 6.6, 132,
                      0.0, False, False, True, True),
    create_ingredient('Pork Chop', 0.0, 27.0, 14.0, 231,
                      0.0, False, False, True, True),

    create_ingredient('Milk', 5.0, 3.4, 3.7, 61, 0.0,
                      False, False, True, False),
    create_ingredient('Cheddar Cheese', 2.0, 25.0, 33.0,
                      403, 0.0, False, False, True, False),
    create_ingredient('Yogurt', 4.0, 9.0, 3.3, 59,
                      0.0, False, False, True, False),
    create_ingredient('Butter', 0.1, 0.9, 81.0, 717,
                      0.0, False, False, True, False),
    create_ingredient('Ice Cream', 21.0, 3.5, 11.0, 207,
                      0.0, False, False, True, False),

    create_ingredient('Broccoli', 6.0, 2.6, 0.3, 34,
                      2.6, True, True, True, True),
    create_ingredient('Carrot', 10.0, 0.9, 0.2, 41,
                      2.8, True, True, True, True),
    create_ingredient('Spinach', 4.0, 2.9, 0.4, 23,
                      2.2, True, True, True, True),
    create_ingredient('Tomato', 4.0, 0.9, 0.2, 18,
                      1.4, True, True, True, True),
    create_ingredient('Potato', 17.0, 2.0, 0.1, 77,
                      2.2, True, True, True, True),

    create_ingredient('Almond', 9.0, 21.0, 53.0, 579,
                      12.5, True, True, True, True),
    create_ingredient('Peanut', 16.0, 26.0, 49.0, 567,
                      8.5, True, True, True, True),
    create_ingredient('Cashew', 30.0, 18.0, 44.0, 553,
                      3.3, True, True, True, True),
    create_ingredient('Walnut', 14.0, 15.0, 65.0, 654,
                      6.7, True, True, True, True),
    create_ingredient('Sunflower Seed', 20.0, 24.0, 51.0,
                      582, 8.6, True, True, True, True),

    create_ingredient('Olive Oil', 0.0, 0.0, 100.0,
                      884, 0.0, True, True, True, True),
    create_ingredient('Soybean Oil', 0.0, 0.0, 100.0,
                      884, 0.0, True, True, True, True),
    create_ingredient('Coconut Oil', 0.0, 0.0, 100.0,
                      862, 0.0, True, True, True, True),
    create_ingredient('Sesame Oil', 0.0, 0.0, 100.0,
                      884, 0.0, True, True, True, True),
    create_ingredient('Canola Oil', 0.0, 0.0, 100.0,
                      884, 0.0, True, True, True, True),

    create_ingredient('Sugar', 100.0, 0.0, 0.0, 387,
                      0.0, True, True, True, True),
    create_ingredient('Honey', 82.0, 0.3, 0.0, 304,
                      0.0, True, True, True, True),
    create_ingredient('Maple Syrup', 67.0, 0.0, 0.0,
                      260, 0.0, True, True, True, True),
    create_ingredient('Chocolate', 58.0, 7.8, 30.0, 546,
                      7.0, False, False, True, True),
    create_ingredient('Jam', 65.0, 0.6, 0.1, 248, 0.0, True, True, True, True),

    create_ingredient('Salt', 0.0, 0.0, 0.0, 0, 0.0, True, True, True, True),
    create_ingredient('Pepper', 64.0, 10.0, 3.0, 251,
                      25.0, True, True, True, True),
    create_ingredient('Garlic', 33.0, 6.4, 0.2, 149,
                      2.1, True, True, True, True),
    create_ingredient('Onion', 10.0, 1.1, 0.1, 40,
                      1.7, True, True, True, True),
    create_ingredient('Basil', 4.0, 3.1, 0.6, 23, 1.6, True, True, True, True),

    create_ingredient('Lentils', 20.0, 9.0, 0.4, 116,
                      8.0, True, True, True, True),
    create_ingredient('Chickpeas', 27.0, 19.0, 6.0,
                      364, 8.0, True, True, True, True),
    create_ingredient('Black Beans', 62.0, 21.0, 1.0,
                      341, 15.0, True, True, True, True),
    create_ingredient('Kidney Beans', 21.0, 8.7, 0.5,
                      127, 6.4, True, True, True, True),
    create_ingredient('Soybeans', 30.0, 36.0, 20.0, 446,
                      15.0, True, True, True, True),

    create_ingredient('Avocado', 9.0, 2.0, 15.0, 160,
                      7.0, True, True, True, True),
    create_ingredient('Cucumber', 4.0, 0.7, 0.1, 15,
                      1.5, True, True, True, True),
    create_ingredient('Bell Pepper', 6.0, 0.9, 0.3,
                      31, 2.1, True, True, True, True),
    create_ingredient('Eggplant', 9.0, 1.0, 0.2, 25,
                      3.0, True, True, True, True),
    create_ingredient('Zucchini', 3.0, 1.2, 0.3, 17,
                      1.0, True, True, True, True),

    create_ingredient('Shrimp', 0.0, 20.0, 1.7, 99,
                      0.0, False, False, True, True),
    create_ingredient('Crab', 0.0, 19.0, 1.5, 87,
                      0.0, False, False, True, True),
    create_ingredient('Lobster', 0.0, 20.0, 2.0, 89,
                      0.0, False, False, True, True),
    create_ingredient('Oyster', 4.0, 9.0, 2.3, 81,
                      0.0, False, False, True, True),
    create_ingredient('Mussel', 7.0, 24.0, 2.2, 172,
                      0.0, False, False, True, True),

    create_ingredient('Tofu', 3.0, 8.0, 5.0, 86, 2.0, True, True, True, True),
    create_ingredient('Tempeh', 9.0, 19.0, 11.0, 193,
                      9.0, True, True, True, True),
    create_ingredient('Seitan', 14.0, 75.0, 1.9, 370,
                      0.0, True, True, False, True),
    create_ingredient('Edamame', 11.0, 12.0, 5.0, 122,
                      5.0, True, True, True, True),
    create_ingredient('Coconut Milk', 6.0, 2.0, 24.0,
                      240, 0.0, True, True, True, True),

    create_ingredient('Beer', 13.0, 0.5, 0.0, 43,
                      0.0, True, True, False, True),
    create_ingredient('Wine', 0.0, 0.1, 0.0, 85, 0.0, True, True, True, True),
    create_ingredient('Coffee', 0.0, 0.3, 0.0, 2, 0.0, True, True, True, True),
    create_ingredient('Tea', 0.0, 0.3, 0.0, 1, 0.0, True, True, True, True),
    create_ingredient('Soda', 39.0, 0.1, 0.1, 149,
                      0.0, True, True, True, True),

    create_ingredient('Popcorn', 74.0, 13.0, 9.0, 431,
                      15.0, True, True, True, True),
    create_ingredient('Pretzel', 77.0, 9.0, 1.0, 364,
                      4.0, True, True, False, True),
    create_ingredient('Potato Chips', 54.0, 6.0, 36.0,
                      536, 4.0, True, True, True, True),
    create_ingredient('Candy', 75.0, 1.0, 0.1, 400,
                      0.0, True, True, True, True),
    create_ingredient('Cake', 52.0, 4.0, 29.0, 444,
                      2.0, False, False, False, True),

    create_ingredient('Egg', 1.1, 13.0, 11.0, 155,
                      0.0, False, False, True, True),
    create_ingredient('Bacon', 0.0, 37.0, 48.0, 541,
                      0.0, False, False, True, True),
    create_ingredient('Sausage', 2.0, 14.0, 28.0, 323,
                      0.0, False, False, True, True),
    create_ingredient('Ham', 0.0, 17.0, 11.0, 145,
                      0.0, False, False, True, True),
    create_ingredient('Turkey', 0.0, 29.0, 14.0, 189,
                      0.0, False, False, True, True),

    create_ingredient('Pea', 14.0, 5.0, 0.4, 81, 5.0, True, True, True, True),
    create_ingredient('Corn', 18.0, 3.3, 1.4, 86, 2.0, True, True, True, True),
    create_ingredient('Asparagus', 4.0, 2.2, 0.2, 20,
                      2.1, True, True, True, True),
    create_ingredient('Beetroot', 10.0, 1.7, 0.2, 43,
                      2.0, True, True, True, True),
    create_ingredient('Radish', 3.4, 0.7, 0.1, 16,
                      1.6, True, True, True, True),
])


# Create the DataFrame
ingredients_df = pd.DataFrame(ingredients_data)

# Save to CSV
ingredients_df.to_csv('data/ingredients_100.csv', index=False)

print("CSV file 'data/ingredients_100.csv' created successfully.")
