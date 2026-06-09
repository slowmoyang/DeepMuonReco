# MuonIdProducer: Tracker Muon Identification Logic
In this [MuonIdProducer](https://github.com/cms-sw/cmssw/blob/CMSSW_20_0_0_pre1/RecoMuon/MuonIdentification/plugins/MuonIdProducer.cc), a **tracker track becomes a CMS tracker muon** only after it is converted into a `reco::Muon`, extrapolated to the muon system, and found to have enough compatible muon-chamber segment matches.

The core predicate is:

```cpp
bool MuonIdProducer::isGoodTrackerMuon(const reco::Muon& muon) {
  if (muon.track()->pt() < minPt_ || muon.track()->p() < minP_)
    return false;

  int numMatches = muon.numberOfMatches(reco::Muon::NoArbitration);

  if (addExtraSoftMuons_ && muon.pt() < 5 && std::abs(muon.eta()) < 1.5 && numMatches >= 1)
    return true;

  return (numMatches >= minNumberOfMatches_);
}
```

So, operationally:

1. **Start from an inner tracker track.**
   The code loops over `innerTrackCollectionHandle_`, usually `generalTracks`, and first applies `isGoodTrack(track)`: minimum `pt`, momentum/`p`, and `|eta|` cuts. If `selectHighPurity_` is enabled, non-`highPurity` tracks are rejected unless the first PV is fake.

2. **Build a temporary `reco::Muon` with the tracker track as `InnerTrack`.**
   The candidate is created with `makeMuon(..., reco::Muon::InnerTrack)`, so the tracker track becomes the muon’s `track()` / `innerTrack()`.

3. **Extrapolate the track to the muon system and fill chamber/segment matches.**
   `fillMuonId()` calls `trackAssociator_.associate(...)`, then fills `MuonChamberMatch` and segment matches. A segment is accepted only if both local x and local y match the propagated track position, either by absolute windows or by pull cuts:
   `|dx| < maxAbsDx` or pull-x criterion, and `|dy| < maxAbsDy` or pull-y criterion. If both pass, the segment is stored as a DT/CSC segment match, GEM match, or ME0 match depending on the chamber type.

4. **Apply `isGoodTrackerMuon()`.**
   The candidate is a tracker muon if:

   * `track.pt() >= minPt_`
   * `track.p() >= minP_`
   * and `muon.numberOfMatches(reco::Muon::NoArbitration) >= minNumberOfMatches_`

   There is a special extra-soft option: if `addExtraSoftMuons_` is true, then for `pt < 5 GeV`, `|eta| < 1.5`, at least one match is enough.

5. **Set the `TrackerMuon` type bit.**
   If `isGoodTrackerMuon(trackerMuon)` returns true, the code sets
   `trackerMuon.setType(trackerMuon.type() | reco::Muon::TrackerMuon)`. If a global muon with the same inner track already exists, it updates that existing muon and ORs in the `TrackerMuon` bit.

6. **After arbitration, the bit can be removed.**
   With the default `arbitrateTrackerMuons = True`, the producer later arbitrates shared segments. If the tracker muon has fewer than `minNumberOfMatches_` after `SegmentAndTrackArbitration` plus GEM arbitration, it either removes a pure tracker-muon candidate or clears the `TrackerMuon`/`GEMMuon` bits on a non-pure muon.

For the default `muons1stStep` configuration in this tag, the relevant values are typically `minPt = 0.5`, `minP = 2.5`, `maxAbsEta = 3.0`, `maxAbsDx = 3.0`, `maxAbsPullX = 3.0`, `minNumberOfMatches = 1`, `addExtraSoftMuons = False`, and `arbitrateTrackerMuons = True`.

So the practical test for an already-produced CMS muon collection is:

```cpp
bool isThisTrackATrackerMuon = false;

for (const auto& mu : *muons) {
  if (mu.innerTrack().isNonnull() &&
      mu.innerTrack().id() == trackRef.id() &&
      mu.innerTrack().key() == trackRef.key() &&
      mu.isTrackerMuon()) {
    isThisTrackATrackerMuon = true;
    break;
  }
}
```

Conceptually: **a tracker track is a tracker muon if it passes the track kinematic/quality preselection and has at least the configured number of compatible muon-system segment matches after track-to-muon-system association and, in the final collection, after arbitration.**
