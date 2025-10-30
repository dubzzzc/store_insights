"""Insights API endpoints.

This module re-exports the sales insights router that dynamically adapts to
Visual FoxPro schema differences (for example, DESCRIPTION vs DESCRIPT).  The
main application continues to import the router from ``app.insights`` so we keep
that import path stable.
"""

from .insights_jnl import router

__all__ = ["router"]
