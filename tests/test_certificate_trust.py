"""Exercise the installed requests trust bundle, without network traffic."""
import hashlib
import ssl

import certifi


# GHSA-6q48-6c7w-9gw2 removed emSign ECC Root CA - G3. Certifi2026.10.19
# deliberately restored the three unaffected emSign roots; blocking the whole
# organization would break their valid certificate chains. See upstream commit
# 662df63d7906df66f1f07c0751a3dfe9da14d914 and the Python security announcement.
EMSIGN_ECC_G3 = "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"

UNAFFECTED_EMSIGN_ROOTS = {
    "40f6af0346a99aa1cd1d555a4e9cce62c7f9634603ee406615833dc8c8d00367",  # G1
    "125609aa301da0a249b97a8239cb6a34216f44dcac9f3954b14292f2e8c8608f",  # C1
    "bc4d809b15189d78db3e1d8cf4f9726a795da1643ca5f1358e1ddb0edc0d7eb3",  # C3
}


def test_requests_bundle_excludes_distrusted_emsign_roots():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=certifi.where())
    roots = context.get_ca_certs(binary_form=True)
    assert roots, "The trust bundle must still contain valid CA roots"
    fingerprints = {hashlib.sha256(root).hexdigest() for root in roots}
    assert EMSIGN_ECC_G3 not in fingerprints, "Distrusted emSign ECC G3 remains in certifi"
    assert UNAFFECTED_EMSIGN_ROOTS <= fingerprints, "Valid emSign roots missing from certifi"
