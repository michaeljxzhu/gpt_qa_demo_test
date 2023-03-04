from langchain.llms import OpenAI

llm = OpenAI(temperature=0.0)

text = """

You are an AI-powered sales assistant who is well-versed in the features and benefits of the product you are selling. The following is a list
of products that you are able to sell.

- vitamin C pills
- protein powder
- multivitamin pills
- matcha green tea powder
- collagen powder
- metabolism enhancement pills
- electrolyte hydration powder
- "Liquid IV" hydration powder
- Skratch Labs energy chews sport fuel
- Powdered Probiotic Kombucha
- isdg 甜蜜习惯抗糖丸 60粒-常驻爆款

Given a customer's question, return the top 3 products relevant to their question in sorted order. If you are not sure of a product's relevance, omit
them from the returned products

Question: Is this suitable for use while running a marathon?
"""

# This ends up working pretty well. It obviously depends on the product description, but the products given to the question are very sensible
# Somewhere in the chain, we will need a part of the prompt to return "no answer"
# print(llm(text))

"""
Question: How many calories does this have?
Answer: Protein powder, Skratch Labs energy chews sport fuel, Powdered Probiotic Kombucha
"""

"""
Question:  I need to lose weight, will this help me?
Answer: Metabolism enhancement pills, electrolyte hydration powder, Skratch Labs energy chews sport fuel
"""

"""
Question: Will this help me be an overall healthier person?

Answer:
1. Multivitamin pills
2. Collagen powder
3. Metabolism enhancement pills
"""

"""
Question: Is this suitable for use while running a marathon?

Answer: Skratch Labs energy chews sport fuel, electrolyte hydration powder, Liquid IV hydration powder
"""

text = """

You are an AI-powered sales assistant who is well-versed in the features and benefits of the product you are selling. Given the question asked by a customer, generate
a 3 to 5 sentence description of a product that is the most relevant to the question.

Question: Is this suitable for use while running a marathon?

"""
print(llm(text))

"""
Question: I need to lose weight, will this help me?

This product is designed to help you reach your weight loss goals. It contains natural ingredients that have been scientifically proven to boost metabolism,
suppress appetite, and increase energy levels. It also provides essential vitamins and minerals to help you stay healthy while you lose weight. Additionally,
it includes a comprehensive diet and exercise plan to help you reach your goals faster. With this product, you can expect to see results in as little as two
weeks.
"""

"""
Question: How many calories does this have?

This product contains 0 calories, making it a great choice for those looking to maintain a healthy diet. It is also free of any artificial sweeteners,
preservatives, and additives, so you can be sure that you are getting a natural and healthy product. Additionally, it is low in fat and sodium, making
it a great choice for those looking to watch their calorie intake.
"""

"""
Question: Will this help me be an overall healthier person?

Yes, this product can help you become an overall healthier person! It contains natural ingredients that are known to boost your immune system, improve digestion,
and provide essential vitamins and minerals. Additionally, it is low in calories and sugar, so you can enjoy it without worrying about your health. Finally, it
is packed with antioxidants that can help protect your body from the damaging effects of free radicals.
"""

"""
Question: Is this suitable for use while running a marathon?

Yes, this product can help you become an overall healthier person! It contains natural ingredients that are known to boost your immune system, improve digestion,
and provide essential vitamins and minerals. Additionally, it is low in calories and sugar, so you can enjoy it without worrying about your health. Finally, it
is packed with antioxidants that can help protect your body from the damaging effects of free radicals.
"""
