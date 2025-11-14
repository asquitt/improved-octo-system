#!/usr/bin/env python3
"""
Track your learning progress through the course

Usage:
    python scripts/check_progress.py
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def check_week_completion(week_num):
    """Check if a week is completed"""
    week_dir = BASE_DIR / f"week-{week_num}"

    if not week_dir.exists():
        return "⬜ Not Available"

    # Check if starter code exists and has been modified
    starter_code = week_dir / "starter_code.py"
    if not starter_code.exists():
        return "⬜ Not Started"

    # Simple check: if file is bigger than template, assume work done
    size = starter_code.stat().st_size
    if size < 1000:  # Very small = not started
        return "⬜ Not Started"
    elif size < 5000:  # Some work
        return "🔄 In Progress"
    else:  # Substantial work
        return "✅ Complete"


def main():
    """Display progress"""
    print("\n" + "="*60)
    print(" "*15 + "📊 LEARNING PROGRESS")
    print("="*60 + "\n")

    weeks = [
        (1, "Foundation"),
        (2, "Data Parallel (DDP)"),
        (3, "Model Parallel"),
        (4, "FSDP"),
        (5, "Advanced Optimizations"),
        (6, "Best Practices"),
        (7, "Deployment & Scaling"),
        (8, "Capstone Project"),
    ]

    total = len(weeks)
    completed = 0

    for week_num, title in weeks:
        status = check_week_completion(week_num)
        if "Complete" in status:
            completed += 1

        print(f"Week {week_num}: {title:<30} {status}")

    print("\n" + "="*60)
    percentage = (completed / total) * 100
    print(f"Overall Progress: {completed}/{total} weeks ({percentage:.0f}%)")

    # Progress bar
    bar_length = 40
    filled = int(bar_length * completed / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\n[{bar}] {percentage:.0f}%")

    print("\n" + "="*60)

    # Recommendations
    if completed == 0:
        print("\n💡 Get started with Week 1:")
        print("   cd week-1")
        print("   python starter_code.py")
    elif completed < total:
        next_week = completed + 1
        print(f"\n💡 Continue with Week {next_week}:")
        print(f"   cd week-{next_week}")
        print("   cat README.md")
    else:
        print("\n🎉 Congratulations! You've completed the entire course!")
        print("   Now apply your skills to real projects!")

    print()


if __name__ == '__main__':
    main()
