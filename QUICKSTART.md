# ðŸš€ í€µ ìŠ¤íƒ€íŠ¸ ê°€ì´ë“œ

## 1ë‹¨ê³„: í™˜ê²½ ì„¤ì •

### Python ê°€ìƒí™˜ê²½ ìƒì„±
```bash
python -m venv trading_env
source trading_env/bin/activate  # Linux/Mac
# trading_env\Scripts\activate  # Windows
```

### í•„ìˆ˜ ë¼ì´ë¸ŒëŸ¬ë¦¬ ì„¤ì¹˜
```bash
pip install -r requirements.txt
```

## 2ë‹¨ê³„: API í‚¤ ì„¤ì •

### .env íŒŒì¼ ìƒì„±
```bash
cp .env.template .env
nano .env  # ë˜ëŠ” í…ìŠ¤íŠ¸ ì—ë””í„°ë¡œ íŽ¸ì§‘
```

### í•„ìˆ˜ API í‚¤ (ìµœì†Œ ì„¤ì •)
```env
# ê¸°ë³¸ ì„¤ì •
INITIAL_BALANCE=50000

# í…”ë ˆê·¸ëž¨ (í•„ìˆ˜)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Twitter (ê¶Œìž¥)
TWITTER_BEARER_TOKEN=your_twitter_token

# Reddit (ê¶Œìž¥) 
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
```

## 3ë‹¨ê³„: ì‹œìŠ¤í…œ í…ŒìŠ¤íŠ¸

### ì•Œë¦¼ ì‹œìŠ¤í…œ í…ŒìŠ¤íŠ¸
```bash
python -c "
from src.notifications import NotificationManager
nm = NotificationManager()
nm.test_notifications()
"
```

### ë°ì´í„° ìˆ˜ì§‘ í…ŒìŠ¤íŠ¸
```bash
python -c "
from src.data_manager import MultiCoinDataManager
dm = MultiCoinDataManager()
prices = dm.get_coin_prices(['BTC', 'ETH'])
print('ê°€ê²© ì¡°íšŒ ì„±ê³µ:', prices)
"
```

## 4ë‹¨ê³„: ì‹¤í–‰

### ë°±í…ŒìŠ¤íŠ¸ ëª¨ë“œ (ì•ˆì „)
```bash
# config/settings.pyì—ì„œ ì‹œë®¬ë ˆì´ì…˜ ëª¨ë“œ í™•ì¸
python main.py --backtest
```

### ì‹¤ì‹œê°„ ëª¨ë“œ
```bash
python main.py
```

## ðŸ“Š ëª¨ë‹ˆí„°ë§

### ë¡œê·¸ ì‹¤ì‹œê°„ í™•ì¸
```bash
tail -f logs/trading.log
```

### í…”ë ˆê·¸ëž¨ ì•Œë¦¼ í™•ì¸
- ë´‡ ì‹œìž‘ ì•Œë¦¼
- ê±°ëž˜ ì‹ í˜¸ ì•Œë¦¼
- ì¼ì¼ ë¦¬í¬íŠ¸ (ì˜¤í›„ 6ì‹œ)

## âš ï¸ ì•ˆì „ ìˆ˜ì¹™

1. **ì†Œì•¡ìœ¼ë¡œ ì‹œìž‘**: `INITIAL_BALANCE=10000` (1ë§Œì›)
2. **ë°±í…ŒìŠ¤íŠ¸ ë¨¼ì €**: ì‹¤ì œ ìžê¸ˆ íˆ¬ìž… ì „ ì¶©ë¶„í•œ í…ŒìŠ¤íŠ¸
3. **ì†ì‹¤ í•œë„ ì„¤ì •**: ì¼ì¼/ì›”ê°„ ìµœëŒ€ ì†ì‹¤ í•œë„
4. **ì •ê¸° ëª¨ë‹ˆí„°ë§**: ì¼ì¼ ì„±ê³¼ í™•ì¸

## ðŸ”§ ì£¼ìš” ì„¤ì •

### í¬íŠ¸í´ë¦¬ì˜¤ ë°°ë¶„ ìˆ˜ì •
`config/settings.py`:
```python
TARGET_ALLOCATION = {
    'BTC': 0.40,    # 40%
    'ETH': 0.30,    # 30% 
    'XRP': 0.10,    # 10%
    'CASH': 0.20    # 20%
}
```

### ê±°ëž˜ ì „ëžµ ì¡°ì •
```python
TRADING_CONFIG = {
    'buy_threshold': 0.7,      # ë” ë³´ìˆ˜ì 
    'sell_threshold': 0.7,
    'stop_loss': 0.03,         # 3% ì†ì ˆ
    'take_profit': 0.08        # 8% ìµì ˆ
}
```

## ðŸ†˜ ë¬¸ì œ í•´ê²°

### ìžì£¼ ë°œìƒí•˜ëŠ” ì˜¤ë¥˜

**ëª¨ë“ˆ ì—†ìŒ ì˜¤ë¥˜**
```bash
pip install pandas numpy requests python-dotenv
```

**API ì—°ê²° ì‹¤íŒ¨**
```bash
# .env íŒŒì¼ API í‚¤ ìž¬í™•ì¸
# ë„¤íŠ¸ì›Œí¬ ì—°ê²° ìƒíƒœ í™•ì¸
```

**ê¶Œí•œ ì˜¤ë¥˜** 
```bash
chmod +x main.py
mkdir logs data backups
```

## ðŸ’¡ íŒ

1. **ë””ìŠ¤ì½”ë“œ/í…”ë ˆê·¸ëž¨ ì±„ë„**: ë‹¤ë¥¸ ì‚¬ìš©ìžë“¤ê³¼ ì •ë³´ ê³µìœ 
2. **ë¡œê·¸ ëª¨ë‹ˆí„°ë§**: ì‹œìŠ¤í…œ ìƒíƒœ ì£¼ê¸°ì  í™•ì¸  
3. **ì„¤ì • ë°±ì—…**: `.env` íŒŒì¼ ì•ˆì „í•œ ê³³ì— ë°±ì—…
4. **ì—…ë°ì´íŠ¸**: ì •ê¸°ì ìœ¼ë¡œ ìµœì‹  ë²„ì „ í™•ì¸

---
**ì‹œìž‘ì´ ì ˆë°˜**: ì†Œì•¡ìœ¼ë¡œ ì‹œìž‘í•´ì„œ ê²½í—˜ì„ ìŒ“ì•„ê°€ì„¸ìš”! ðŸš€