/**
 * Browser Notification Service
 * Handles notification permissions and sending alerts for high-confidence signals.
 */

export type NotificationPermission = 'granted' | 'denied' | 'default';

interface NotificationOptions {
    title: string;
    body: string;
    icon?: string;
    tag?: string;
    requireInteraction?: boolean;
}

class NotificationService {
    private permission: NotificationPermission = 'default';
    private enabled: boolean = false;

    constructor() {
        if ('Notification' in window) {
            this.permission = Notification.permission;
            this.enabled = localStorage.getItem('notifications_enabled') === 'true';
        }
    }

    async requestPermission(): Promise<NotificationPermission> {
        if (!('Notification' in window)) {
            console.warn('Notifications not supported');
            return 'denied';
        }

        try {
            this.permission = await Notification.requestPermission();
            if (this.permission === 'granted') {
                this.enabled = true;
                localStorage.setItem('notifications_enabled', 'true');
            }
            return this.permission;
        } catch (error) {
            console.error('Failed to request notification permission:', error);
            return 'denied';
        }
    }

    isSupported(): boolean {
        return 'Notification' in window;
    }

    isEnabled(): boolean {
        return this.enabled && this.permission === 'granted';
    }

    setEnabled(enabled: boolean): void {
        this.enabled = enabled;
        localStorage.setItem('notifications_enabled', enabled.toString());
    }

    getPermission(): NotificationPermission {
        return this.permission;
    }

    send(options: NotificationOptions): Notification | null {
        if (!this.isEnabled()) {
            return null;
        }

        try {
            const notification = new Notification(options.title, {
                body: options.body,
                icon: options.icon || '/favicon.ico',
                tag: options.tag,
                requireInteraction: options.requireInteraction ?? false,
            });

            notification.onclick = () => {
                window.focus();
                notification.close();
            };

            // Auto-close after 5 seconds
            setTimeout(() => notification.close(), 5000);

            return notification;
        } catch (error) {
            console.error('Failed to send notification:', error);
            return null;
        }
    }

    // Convenience method for high-confidence signal alerts
    sendSignalAlert(ticker: string, action: string, confidence: number): void {
        if (confidence < 0.8) return; // Only notify for high-confidence signals

        this.send({
            title: `🎯 ${ticker} - ${action}`,
            body: `Confidence: ${Math.round(confidence * 100)}%`,
            tag: `signal-${ticker}`,
            requireInteraction: confidence >= 0.9,
        });
    }

    // Alert when portfolio threshold is breached
    sendPortfolioAlert(type: 'profit' | 'loss', percentage: number): void {
        const emoji = type === 'profit' ? '📈' : '📉';
        this.send({
            title: `${emoji} Portfolio ${type === 'profit' ? 'Up' : 'Down'} ${Math.abs(percentage).toFixed(1)}%`,
            body: 'Open Signal.Engine to review',
            tag: 'portfolio-alert',
            requireInteraction: true,
        });
    }
}

// Singleton instance
export const notificationService = new NotificationService();
