from mercilt4_client import MercilClient
import json

# 1. ใส่ Key ของคุณ (ต้องตรงกับใน .env)
API_KEY = "mer-GkI78nqrLJVTlbGDZ0C85PfAxIEIeWvoYPzZpGXEFHAx5ar7"

# 2. เชื่อมต่อ
client = MercilClient(api_key=API_KEY)

# 3. ทดสอบค้นหา (แบบไม่ใส่ Filter ราคา/จังหวัด)
# แค่ใส่ query อย่างเดียว ระบบจะส่ง filters={} ไปให้เอง
query_text = "หาใกล้รถไฟฟ้าระยะไม่เกิน 2 กิโลเมตร"

print(f"🔍 กำลังทดสอบ AI ด้วย Query: '{query_text}'\n")
response = client.search(query=query_text)

# 4. ดูผลลัพธ์การจัดอันดับ
if response and "results" in response:
    print(f"✅ เจอทั้งหมด: {len(response['results'])} รายการ\n")
    
    for i, item in enumerate(response['results']):
        details = item['asset_details']
        print(f"อันดับ {i+1}: {details['name']}")
        print(f"   💰 ราคา: {details['price']:,.0f} บาท")
        print(f"   ⭐ คะแนนรวม (Final Score): {item['final_score']}")
        print(f"   🧠 คะแนนความตรงใจ (Intent Score): {item['intent_score']}")
        print(f"   🤖 AI สรุป: {item['summary']}")
        print("-" * 50)
else:
    print("❌ ไม่พบข้อมูล หรือเกิดข้อผิดพลาด")