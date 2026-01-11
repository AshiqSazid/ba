
### **Running the Server**
Python 3.12
Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```
http://localhost:8000
```

```
http://localhost:8000/api/recommendations
http://localhost:8000/api/feedback
http://localhost:8000/api/export
