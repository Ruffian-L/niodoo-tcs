import unittest
from unittest.mock import MagicMock
from organized.ai.EchoMemoria.core.heart import HeartCore

class TestHeartIntegration(unittest.TestCase):
    def test_heart_integration(self):
        event_bus = MagicMock()
        state_store = MagicMock()
        decision_engine = MagicMock()

        heart = HeartCore(event_bus, state_store, decision_engine)
        heart.start()

        self.assertEqual(event_bus.subscribe.call_count, 3)
        event_bus.subscribe.assert_any_call("perception.signal", heart.route_signal)
        event_bus.subscribe.assert_any_call("decision.request", heart.choose_intent)
        event_bus.subscribe.assert_any_call("heartbeat", heart.emit_guidance)

if __name__ == '__main__':
    unittest.main()
