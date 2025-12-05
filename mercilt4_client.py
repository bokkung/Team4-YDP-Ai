import requests
import os

class MercilClient:
    
    def __init__(self, api_key: str = None, base_url: str = "http://127.0.0.1:8000"):
        
        self.api_key = api_key or os.getenv("MERCIL_API_KEY")
        self.base_url = base_url.rstrip("/")
        
        if not self.api_key:
            raise ValueError(" กรุณาใส่ API Key หรือตั้งค่าใน .env")

    def search(self, query: str, max_price: int = None, province: str = None):
        endpoint = f"{self.base_url}/api/v1/search"
        
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        
        filters = {}
        if max_price: filters["max_price"] = max_price
        if province: filters["province"] = province
        
        payload = {
            "query": query,
            "filters": filters
        }
        
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers)
            response.raise_for_status() 
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f" เชื่อมต่อ API ไม่สำเร็จ: {e}")
            if e.response:
                print(f"Server ตอบว่า: {e.response.text}")
            return None