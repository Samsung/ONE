# How to Contribute

**ONE** always welcomes your contribution, but there are basic guidelines that you should follow in
order to make your contribution be accepted.

This document explains such guidelines for beginners.

## General contribution guidelines

If you are not familiar with git or github, please visit
[here](https://guides.github.com/activities/hello-world/) for basic understanding of git and github.

## How to create a Pull Request

This section explains the steps to create a pull request (PR).

1. Create an issue

   Maintainers will accept your contribution only when it is well aligned with the 
   [roadmap](https://github.com/Samsung/ONE/blob/master/docs/overview/roadmap.md) and design
   principles of **ONE**. So, it is optional, but recommended for contributors to create an issue
   and have a discussion with maintainers before writing code.

1. Create a draft PR

   Maintainers will accept your pull request only when it is **reasonably small** and **focused**.
   Sometimes, your contribution may require huge and loosely-coupled changes. You **should** split
   your contribution into multiple small, but focused pull requests in this case. Unfortunately, it
   is possible that maintainers reject your pull request as it is hard for them to understand the
   intuition behind these changes. So, it is optional, but recommended for contributors to present
   the full [draft](https://github.com/Samsung/ONE/pulls?q=is%3Apr+label%3ADRAFT+) of your 
   contribution and have a discussion with maintainers before creating PR(s).

1. Create a commit

   It is time to create a commit for submission once you are convinced that your contribution is
   ready to go. Please include signed-off message at the end of commit message. If not, your pull
   request will be **rejected** by CI.

1. Check code format locally

   **ONE** has its code formatting rules, and any pull request that violates these rules will be
   **rejected** by CI. So, it is optional, but recommended for contributor to check code format
   locally before submission.

1. Create a PR

   It is time to send a pull request. Please explain your intention via description. Maintainers
   will review your pull request based on that description. Each pull request needs approval from at
   least two reviewers to be accepted. Note that **description should include at least four words**.
   If not, your pull request will be **rejected** by CI.

1. Request review

   It is recommended to assign reviewers yourself. Maintainers will honor your review request,
   and accept your pull request only when

   - Approved by 1+ reviewers
   - 0 rejection(Request Changes)
   - 0 pending review request
   - All the reviewers in the list must approve your pull request

   You can add/remove pending review requests in the middle of the review process. Maintainers
   (or reviewers) could review your pull request even without explicit review request.

1. Update per feedback

   Sometimes, maintainers (or reviewers) will request changes on your pull request. Please update
   your pull request upon such feedbacks. These update commits will be squashed into the first
   commit of your pull request later. Please do **NOT** include a sign-off message or write a full
   description for update commits.

