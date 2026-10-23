"""Exercise the installed requests trust bundle, without network traffic."""
import hashlib
import ssl

import certifi


# GHSA-6q48-6c7w-9gw2 removed emSign ECC Root CA - G3. Certifi2026.10.19
# deliberately restored the three unaffected emSign roots; blocking the whole
# organization would break their valid certificate chains. See upstream commit
# 662df63d7906df66f1f07c0751a3dfe9da14d914 and the Python security announcement.
EMSIGN_ECC_G3 = "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"


def test_requests_bundle_excludes_distrusted_emsign_roots():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=certifi.where())
    roots = context.get_ca_certs(binary_form=True)
    assert roots, "The trust bundle must still contain valid CA roots"
    fingerprints = {hashlib.sha256(root).hexdigest() for root in roots}
    assert EMSIGN_ECC_G3 not in fingerprints, "Distrusted emSign ECC G3 remains in certifi"
