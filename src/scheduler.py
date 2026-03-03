import logging
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger("Scheduler")

# Global scheduler instance
scheduler = AsyncIOScheduler()

def start_scheduler(scan_function):
    """
    Initializes and starts the APScheduler.
    Runs the scan_function autonomously every 5 minutes during trading hours.
    Allows the trading agent to operate regardless of active frontend websocket connections.
    """
    # Remove any existing jobs to prevent duplicates on reload
    scheduler.remove_all_jobs()
    
    # Schedule the scan to run every 5 minutes, Monday through Friday
    # Assuming Indian standard hours roughly 9am-4pm
    scheduler.add_job(
        scan_function,
        CronTrigger(day_of_week='mon-fri', hour='9-15', minute='*/5'),
        id="autonomous_market_scan",
        name="Market Scanner",
        replace_existing=True
    )
    
    # You can also add a background job that runs right at startup
    scheduler.add_job(
        scan_function,
        'date',
        id="startup_scan",
        name="Boot Initial Scanner"
    )

    scheduler.start()
    logger.info("✅ APScheduler autonomous scanner engaged.")

def stop_scheduler():
    """Gracefully shuts down the background scanner."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("🛑 APScheduler autonomous scanner shut down.")
