"""Adversarial grounding cases and valid controls found during the QA review."""
import pytest

from app.evaluator import evaluate_relevance
from app.generator import check_quantity_support, extract_quantities, verify_citations_grounding
from tests.test_answer_regressions import evidence


def test_partially_overlapping_metric_cannot_borrow_quantity():
    source = 'Response time is 4 hours. Response policy is 30 days.'
    assert not check_quantity_support('Response time is 30 days.', source)[0]
    assert not verify_citations_grounding('Response time is 30 days [1].', [evidence(source)])['all_supported']
    assert check_quantity_support('Response time is 4 hours.', source)[0]
    assert check_quantity_support('Response policy is 30 days.', source)[0]


@pytest.mark.parametrize('claim,source,expected', [
    ('30 consecutive days', '30 consecutive hours', False),
    ('30 calendar days', '30 business days', False),
    ('30 working days', '30 business days', True),
    ('30 hrs', '30 hours', True),
    ('30 consecutive days', '30 days', True),
])
def test_duration_units_remain_strict(claim, source, expected):
    assert extract_quantities(claim)[0][1] is not None
    assert check_quantity_support('Response time is ' + claim, 'Response time is ' + source)[0] is expected


@pytest.mark.parametrize('query', [
    'What is uptime target?', 'What is the uptime target?',
    'What is availability goal?', 'What is the uptime?',
    'What is the support response time?', 'What is the notice period?',
    'What is the price?', 'What is the storage limit?',
])
def test_value_queries_require_a_value(query):
    source = 'Uptime target, availability goal, support response time, notice period, price and storage limit are under review.'
    assert evaluate_relevance([evidence(source)], query)['status'] == 'REFUSE'


@pytest.mark.parametrize('query,source', [
    ('What is uptime?', 'Uptime describes the time a service is operating.'),
    ('What is availability?', 'Availability describes whether a service is accessible.'),
    ('Define an uptime target.', 'An uptime target is the desired level of service reliability.'),
    ('What does availability goal mean?', 'An availability goal means the desired service accessibility.'),
    ('What is the support response time?', 'The support response time is 4 hours.'),
    ('What is the uptime target?', 'The uptime target is 99.9%.'),
])
def test_definitions_and_supported_values_still_work(query, source):
    assert evaluate_relevance([evidence(source)], query)['status'] == 'SYNTHESIZE'


def test_evaluator_does_not_borrow_a_related_metrics_percentage():
    source = 'Service uptime is not specified. Service discount is 30%.'
    assert evaluate_relevance([evidence(source)], 'What is the service uptime?')['status'] == 'REFUSE'


def test_comma_grouped_quantity_keeps_its_statement():
    assert check_quantity_support('Liability cap is $50,000.', 'Liability cap is $50,000.')[0]
