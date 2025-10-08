import asyncio
import random
from playwright.async_api import async_playwright

async def open_url_and_fill():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://x.com/?lang=en")

        sign_in_button = await page.wait_for_selector(
            'xpath=//*[@id="react-root"]/div/div/div[2]/main/div/div/div[1]/div[1]/div/div[3]/div[4]/a/div/span/span',
            timeout=30000
        )
        await sign_in_button.click()

        input_element = await page.wait_for_selector(
            'xpath=//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[4]/label/div/div[2]/div/input',
            timeout=30000
        )

        contact_number = "9815767797"
        for digit in contact_number:
            await input_element.type(digit, delay=random.randint(100, 400))

        print("Filled contact number:", contact_number)

        next_button = await page.wait_for_selector('xpath=//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/button[2]/div')
        await next_button.click()

        await page.wait_for_timeout(3000)

        input_element = await page.wait_for_selector(
            'xpath=//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div/div[2]/div/label/div/div[2]/div[1]/input',
            timeout=30000
        )

        Password = "Rahul@7355"
        for digit in Password:
            await input_element.type(digit, delay=random.randint(100, 400))

        print("Password", Password)

        await page.locator(
            'xpath=//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div[1]/div/div/button/div/span/span'
        ).click()


        await page.wait_for_timeout(3000)

        search_engine = await page.wait_for_selector('xpath=//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[2]/div/div[2]/div/div/div/div/div[1]/div/div/div/form/div[1]/div/div/div/div/div[2]/div/input')
        await search_engine.fill("#NSE OR #BSE OR $RELIANCE OR $INFY OR #Sensex OR #Nifty")

        await asyncio.sleep(10)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(open_url_and_fill())
