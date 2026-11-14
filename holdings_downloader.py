#!/usr/bin/env python3
"""
ETF Holdings Downloader - Downloads ETF holdings from Schwab using Selenium
Integrated from holdings_dl.py with class-based structure
"""

import math
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import os


class ETFHoldingsDownloader:
    """Downloads ETF holdings from Schwab using Selenium (integrated from holdings_dl.py)"""
    
    def __init__(self, output_dir='src/holdings_data', headless=True, wait_time=15):
        """
        Initialize the holdings downloader
        
        Args:
            output_dir (str): Directory to save CSV files
            headless (bool): Run Firefox in headless mode
            wait_time (int): Maximum time to wait for page loads
        """
        self.output_dir = output_dir
        self.wait_time = wait_time
        
        # Initialize Chrome options
        self.chrome_options = Options()
        self.chrome_options.headless = headless
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def _convert_units_to_float(self, x):
        """
        Convert strings like "$53.4B", "7.86%", "288.4M" to floats
        Handles K (thousands), M (millions), B (billions), % (percent)
        """
        start = 0
        if type(x) == float:
            return x
        if x[0] == '-':  # set negative portfolio weights to 0
            return 0
        if x[0] == '$':
            start = 1
        if x[-1] == '%':
            return float(x[:-1]) / 100
        if x[-1] == 'K':
            return float(x[start:-1]) * 10e3
        if x[-1] == 'M':
            return float(x[start:-1]) * 10e6
        if x[-1] == 'B':
            return float(x[start:-1]) * 10e9
        else:
            return float(x[start:])
    
    def _get_etf_from_schwab(self, etf_symbol):
        """
        Selenium scraping logic for single ETF
        Opens Firefox, navigates to Schwab, scrapes all pages
        
        Args:
            etf_symbol (str): ETF symbol to download
            
        Returns:
            bool: True if successful, False if ETF not found
        """
        print(f"Opening {etf_symbol} database")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=self.chrome_options)
        driver.implicitly_wait(self.wait_time)
        wait = WebDriverWait(driver, 30, poll_frequency=1)
        
        url = "https://www.schwab.wallst.com/schwab/Prospect/research/etfs/schwabETF" \
              "/index.asp?type=holdings&symbol={}".format(etf_symbol)
        
        try:
            driver.get(url)
            show_sixty_items = driver.find_element(By.XPATH, "//a[@perpage='60']")
            show_sixty_items.click()
        except ec.NoSuchElementException:
            print(f"{etf_symbol} is not a valid ETF (not found in schwab database)")
            driver.close()
            return False
        except ec.WebDriverException:
            print(f"{etf_symbol} not retrieved (web driver error)")
            driver.close()
            return False
        
        page_elt = wait.until(ec.visibility_of_element_located((By.CLASS_NAME, "paginationContainer")))
        try:
            num_pages = math.ceil(float(page_elt.text.split(" ")[4]) / 60)
        except ec.StaleElementReferenceException:  # fixes stale reference bug
            page_elt = wait.until(ec.visibility_of_element_located((By.CLASS_NAME, "paginationContainer")))
            num_pages = math.ceil(float(page_elt.text.split(" ")[4]) / 60)

        # Read holdings data
        print(f"{etf_symbol}: page 1 of {num_pages} ...", end=" ")
        time.sleep(.5)  # force wait needed for reliability
        dataframe_list = [pd.read_html(driver.page_source)[1]]
        print("complete")
        
        current_page = 2
        while current_page <= num_pages:
            print(f"{etf_symbol}: page {current_page} of {num_pages} ...", end=" ")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            next_button = driver.find_element(By.XPATH, "//li[@pagenumber='{}']".format(current_page))
            driver.execute_script("arguments[0].click();", next_button)
            
            while True:  # wait until the new data has loaded
                time.sleep(.25)
                df = pd.read_html(driver.page_source, match="Symbol")[0]
                if not df.equals(dataframe_list[-1]):
                    break
            dataframe_list.append(df)
            current_page += 1
            print("complete")
        
        # Merge into a single dataframe
        concat_result = pd.concat(dataframe_list)
        result_df = concat_result.drop_duplicates()
        result_df.columns = ['Symbol', 'Description', 'Portfolio Weight', 'Shares Held', 'Market Value']
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"{etf_symbol}-holdings.csv")
        result_df.to_csv(csv_path, index=False)
        
        driver.close()
        print(f"{etf_symbol}: {result_df.shape[0]} holdings retrieved")
        return True
    
    def download_holdings(self, etf_symbols):
        """
        Download holdings for multiple ETFs
        
        Args:
            etf_symbols (list): List of ETF symbols to download
            
        Returns:
            dict: {symbol: csv_path} for successfully downloaded ETFs
        """
        results = {}
        
        for symbol in etf_symbols:
            if self._get_etf_from_schwab(symbol):
                csv_path = os.path.join(self.output_dir, f"{symbol}-holdings.csv")
                results[symbol] = csv_path
                print(f"✅ {symbol}: Downloaded to {csv_path}")
            else:
                print(f"❌ {symbol}: Failed to download")
        
        return results
    
    def check_existing_files(self, etf_symbols):
        """
        Check which ETF CSV files already exist
        
        Args:
            etf_symbols (list): List of ETF symbols to check
            
        Returns:
            dict: {symbol: csv_path} for existing files
        """
        existing = {}
        
        for symbol in etf_symbols:
            csv_path = os.path.join(self.output_dir, f"{symbol}-holdings.csv")
            if os.path.exists(csv_path):
                existing[symbol] = csv_path
                print(f"📁 {symbol}: Already exists at {csv_path}")
        
        return existing
