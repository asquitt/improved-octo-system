#!/usr/bin/env python3
"""Template for complete training loop"""

def train_model(model, train_loader, val_loader, config):
    """
    Complete training pipeline template

    Fill in your custom logic while keeping structure
    """
    # Setup
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = create_criterion(config)

    # Training loop
    for epoch in range(config.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, epoch
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion
        )

        # Step scheduler
        scheduler.step()

        # Checkpoint
        if val_metrics['loss'] < best_loss:
            save_checkpoint(model, optimizer, epoch)

        # Log
        log_metrics(train_metrics, val_metrics, epoch)

    return model
