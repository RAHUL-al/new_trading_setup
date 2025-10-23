import os
from SmartApi import SmartConnect
import pyotp
import sys
from SmartApi.smartConnect import SmartConnect
from concurrent.futures import ThreadPoolExecutor


class TestCases():
    def __init__(self):
        self.api_key = 'SsUDlNA9'
        self.username = 'A1079871'
        self.pwd = '0465'
        self.token = 'OIN6QBZAYV4I26Q55OYASIEQVY'

        totp = pyotp.TOTP(self.token).now()
        self.smart_api = SmartConnect(self.api_key)

        data1 = self.smart_api.generateSession(self.username, self.pwd, totp)
        refreshToken = data1['data']['refreshToken']

        self.smart_api.generateToken(refreshToken)

    def test_ltpdata(self,token_number):
        ltp = self.smart_api.ltpData("NFO", "NIFTY2581424300CE", f"{token_number}")
        assert "data" in ltp
        return ltp

