import argparse
import logging
import psutil
from multiprocessing import resource_tracker

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_shm_from_resource_tracker(shm_name: str):
    """
    Attempts to remove a shared memory segment name from Python's
    multiprocessing.resource_tracker to prevent 'file already exists'
    errors in subsequent runs within the same session.
    
    This is a 'best-effort' operation.
    """
    # Resource tracker names are prefixed with '/'
    tracker_name = f"/{shm_name}"
    try:
        logging.info(f"Attempting to unregister '{tracker_name}' from resource_tracker...")
        resource_tracker.unregister(tracker_name, 'shared_memory')
        logging.info(f"Successfully unregistered '{tracker_name}'.")
    except (KeyError, FileNotFoundError):
        logging.warning(f"'{tracker_name}' not found in resource_tracker, it might have been cleaned up already.")
    except Exception as e:
        logging.error(f"An unexpected error occurred while unregistering '{tracker_name}': {e}")


def find_and_cleanup_ipc(prefix: str, list_only: bool, force: bool):
    """
    Finds and cleans up Boost.Interprocess shared memory and message queues
    that are left orphaned in /dev/shm.
    """
    shm_path = "/dev/shm"
    found_any = False
    
    logging.info(f"Scanning {shm_path} for IPC objects with prefix '{prefix}'...")
    
    try:
        # Boost IPC objects often have prefixes like 'bip.' or 'sem.'
        # We target the names given in our config directly.
        potential_ipc_objects = [f for f in psutil.os.listdir(shm_path) if f.startswith(prefix)]
    except FileNotFoundError:
        logging.error(f"Directory not found: {shm_path}. This script is intended for Linux systems.")
        return
        
    if not potential_ipc_objects:
        logging.info("No orphaned IPC objects found.")
        return

    logging.info(f"Found {len(potential_ipc_objects)} potential orphaned object(s):")
    for item in potential_ipc_objects:
        logging.info(f"  - {item}")
        found_any = True

    if list_only:
        logging.info("List-only mode enabled. No cleanup will be performed.")
        return

    if not force:
        confirm = input("Proceed with cleanup? [y/N]: ")
        if confirm.lower() != 'y':
            logging.info("Cleanup aborted by user.")
            return
            
    logging.info("Proceeding with cleanup...")
    cleaned_count = 0
    for item_name in potential_ipc_objects:
        item_path = psutil.os.path.join(shm_path, item_name)
        try:
            psutil.os.remove(item_path)
            logging.info(f"Removed: {item_path}")
            # Also attempt to unregister from Python's tracker
            remove_shm_from_resource_tracker(item_name)
            cleaned_count += 1
        except OSError as e:
            logging.error(f"Error removing {item_path}: {e}. It might be in use or require higher privileges.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while removing {item_path}: {e}")

    logging.info(f"Cleanup complete. Removed {cleaned_count}/{len(potential_ipc_objects)} objects.")


def main():
    parser = argparse.ArgumentParser(
        description="A-Sys-I IPC Cleanup Tool. Finds and removes orphaned shared memory segments and message queues from /dev/shm.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "prefix",
        type=str,
        help="The prefix of the IPC objects to search for (e.g., 'asys_i_hpc')."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list the found IPC objects without deleting them."
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force cleanup without asking for confirmation."
    )
    
    args = parser.parse_args()
    
    find_and_cleanup_ipc(args.prefix, args.list, args.force)

if __name__ == "__main__":
    main()

