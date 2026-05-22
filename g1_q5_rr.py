"""
Ferropulse Internship Assessment
Group 1 — Question 5: Round Robin Delivery Allocation System

Problem:
    Assign incoming delivery orders to available delivery partners using
    round-robin scheduling. Partners can go online/offline dynamically,
    and orders should only be assigned to currently active partners while
    maintaining fair distribution.

Approach:
    - Maintain an ordered list of active partners as the round-robin queue.
    - A single integer pointer (rr_pointer) tracks the next partner to assign.
    - When a partner goes offline, the pointer is adjusted so the round-robin
      cycle doesn't skip or repeat a position.
    - Time Complexity: O(1) for assignment, O(n) for offline removal (n = active partners).
    - Space Complexity: O(p) where p = total partners ever registered.
"""

from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
#  Data Model
# ──────────────────────────────────────────────

@dataclass
class DeliveryPartner:
    partner_id: str
    name: str
    is_online: bool = False
    orders_assigned: int = 0


# ──────────────────────────────────────────────
#  Core Allocator
# ──────────────────────────────────────────────

class RoundRobinAllocator:
    """
    Manages delivery partner availability and fairly assigns
    incoming orders using round-robin scheduling.
    """

    def __init__(self):
        self._partners: dict[str, DeliveryPartner] = {}   # All known partners
        self._active_queue: list[str] = []                # IDs of online partners in order
        self._rr_pointer: int = 0                         # Next assignment position
        self._order_log: list[tuple[str, str]] = []       # (order_id, partner_id)

    # ── Partner Management ──────────────────────

    def go_online(self, partner_id: str, name: str = "") -> None:
        """Bring a delivery partner online and add them to the rotation."""
        if partner_id not in self._partners:
            self._partners[partner_id] = DeliveryPartner(
                partner_id=partner_id,
                name=name or partner_id
            )

        partner = self._partners[partner_id]
        if partner.is_online:
            print(f"  [INFO] {partner.name} is already online.")
            return

        partner.is_online = True
        self._active_queue.append(partner_id)
        print(f"  🟢 {partner.name} came online  "
              f"[Active partners: {len(self._active_queue)}]")

    def go_offline(self, partner_id: str) -> None:
        """
        Take a delivery partner offline and remove them from the rotation.
        The round-robin pointer is adjusted so the cycle resumes correctly.
        """
        if partner_id not in self._partners or not self._partners[partner_id].is_online:
            print(f"  [INFO] Partner {partner_id} is not currently online.")
            return

        partner = self._partners[partner_id]
        partner.is_online = False

        removed_idx = self._active_queue.index(partner_id)
        self._active_queue.remove(partner_id)

        if self._active_queue:
            # If the removed partner was before the pointer, shift pointer back
            # to avoid skipping the partner now sitting at that position.
            if self._rr_pointer > removed_idx:
                self._rr_pointer -= 1
            # Wrap pointer in case it was pointing at or beyond the end
            self._rr_pointer %= len(self._active_queue)
        else:
            self._rr_pointer = 0

        print(f"  🔴 {partner.name} went offline  "
              f"[Active partners: {len(self._active_queue)}]")

    # ── Order Assignment ─────────────────────────

    def assign_order(self, order_id: str) -> Optional[str]:
        """
        Assign an order to the next partner in the round-robin cycle.
        Returns the partner_id that was assigned, or None if no one is online.
        """
        if not self._active_queue:
            print(f"  ⚠️  Order {order_id} could not be assigned — no active partners!")
            return None

        # Pick current partner and advance pointer
        partner_id = self._active_queue[self._rr_pointer]
        self._rr_pointer = (self._rr_pointer + 1) % len(self._active_queue)

        self._partners[partner_id].orders_assigned += 1
        self._order_log.append((order_id, partner_id))

        print(f"  📦 Order {order_id}  →  {self._partners[partner_id].name}")
        return partner_id

    # ── Reporting ────────────────────────────────

    def print_stats(self) -> None:
        """Display a summary of all partners and their assignments."""
        print("\n" + "═" * 45)
        print("  DELIVERY ALLOCATION SUMMARY")
        print("═" * 45)
        print(f"  {'Partner':<20} {'Orders':>8}   {'Status'}")
        print("  " + "─" * 43)
        for p in self._partners.values():
            status = "🟢 Online " if p.is_online else "🔴 Offline"
            print(f"  {p.name:<20} {p.orders_assigned:>8}   {status}")
        print("═" * 45)
        total_orders = len(self._order_log)
        print(f"  Total orders assigned: {total_orders}")
        if total_orders > 0:
            values = [p.orders_assigned for p in self._partners.values()]
            spread = max(values) - min(v for v in values if v > 0) if any(values) else 0
            print(f"  Max load imbalance:    {spread} order(s)")
        print("═" * 45 + "\n")


# ──────────────────────────────────────────────
#  Demo / Test
# ──────────────────────────────────────────────

def run_demo():
    print("\n" + "═" * 45)
    print("  ROUND ROBIN DELIVERY ALLOCATOR — DEMO")
    print("═" * 45 + "\n")

    allocator = RoundRobinAllocator()

    # ── Phase 1: Three partners come online ──
    print("--- Phase 1: Partners come online ---")
    allocator.go_online("P1", "Ravi")
    allocator.go_online("P2", "Suresh")
    allocator.go_online("P3", "Meena")

    print("\n--- Phase 2: Assign 6 orders (should distribute evenly) ---")
    for i in range(1, 7):
        allocator.assign_order(f"ORD-{i:03d}")

    # ── Phase 3: One partner goes offline ──
    print("\n--- Phase 3: Suresh goes offline ---")
    allocator.go_offline("P2")

    print("\n--- Phase 4: Assign 4 more orders (only Ravi and Meena) ---")
    for i in range(7, 11):
        allocator.assign_order(f"ORD-{i:03d}")

    # ── Phase 5: A new partner joins ──
    print("\n--- Phase 5: New partner Arjun comes online ---")
    allocator.go_online("P4", "Arjun")

    print("\n--- Phase 6: Assign 3 more orders ---")
    for i in range(11, 14):
        allocator.assign_order(f"ORD-{i:03d}")

    # ── Phase 6: No active partners ──
    print("\n--- Phase 7: All partners go offline ---")
    allocator.go_offline("P1")
    allocator.go_offline("P3")
    allocator.go_offline("P4")

    print("\n--- Phase 8: Try to assign order with no active partners ---")
    allocator.assign_order("ORD-014")

    allocator.print_stats()


if __name__ == "__main__":
    run_demo()