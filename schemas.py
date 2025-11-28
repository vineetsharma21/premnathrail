from pydantic import BaseModel

# User jab Signup karega to yeh data bhejega
class UserCreate(BaseModel):
    email: str
    password: str

# User jab Login karega
class UserLogin(BaseModel):
    email: str
    password: str

# User jab License Activate karega
class LicenseActivate(BaseModel):
    license_key: str