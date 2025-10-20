# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from typing import Any

from camel.toolkits import FunctionTool

from oasis.social_platform.channel import Channel
from oasis.social_platform.typing import ActionType


class SocialAction:

    def __init__(self, agent_id: int, channel: Channel):
        self.agent_id = agent_id
        self.channel = channel

    def get_openai_function_list(self) -> list[FunctionTool]:
        return [
            FunctionTool(func) for func in [
                self.create_post,
                self.like_post,
                self.repost,
                self.quote_post,
                self.unlike_post,
                self.dislike_post,
                self.undo_dislike_post,
                self.search_posts,
                self.search_user,
                self.trend,
                self.refresh,
                self.do_nothing,
                self.create_comment,
                self.like_comment,
                self.dislike_comment,
                self.unlike_comment,
                self.undo_dislike_comment,
                self.follow,
                self.unfollow,
                self.mute,
                self.unmute,
                self.purchase_product,
                self.send_private_message,
                self.transfer_money,
                self.click_link,
                self.submit_info,
                self.get_bad_history_conversation,
                self.get_successful_transfers_to_agent, 
                self.get_bad_bad_history_conversation,
            ]
        ]

    async def perform_action(self, message: Any, type: str):
        message_id = await self.channel.write_to_receive_queue(
            (self.agent_id, message, type), self.agent_id)
        response = await self.channel.read_from_send_queue(message_id)
        return response[2]

    async def sign_up(self, user_name: str, name: str, bio: str):
        r"""Signs up a new user with the provided username, name, and bio.

        This method prepares a user message comprising the user's details and
        invokes an asynchronous action to perform the sign-up process. On
        successful execution, it returns a dictionary indicating success along
        with the newly created user ID.

        Args:
            user_name (str): The username for the new user.
            name (str): The full name of the new user.
            bio (str): A brief biography of the new user.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the sign-up was
                successful, and 'user_id' key maps to the integer ID of the
                newly created user on success.

            Example of a successful return:
            {'success': True, 'user_id': 2}
        """

        # print(f"Agent {self.agent_id} is signing up with "
        #       f"user_name: {user_name}, name: {name}, bio: {bio}")
        user_message = (user_name, name, bio)
        return await self.perform_action(user_message, ActionType.SIGNUP.value)

    async def refresh(self):
        r"""Refresh to get recommended posts.

        This method invokes an asynchronous action to refresh and fetch
        recommended posts. On successful execution, it returns a dictionary
        indicating success along with a list of posts. Each post in the list
        contains details such as post ID, user ID, content, creation date,
        and the number of likes.

        Returns:
            dict: A dictionary with three key-value pairs. The 'success' key
                maps to a boolean indicating whether the refresh is
                successful. The 'posts' key maps to a list of dictionaries,
                each representing a post with its details. The 'private_messages' key maps to a list of dictionaries,

            Example of a successful return:
            {
                "success": True,
                "posts": [
                    {
                        "post_id": 1,
                        "user_id": 23,
                        "content": "This is an example post content.",
                        "created_at": "2024-05-14T12:00:00Z",
                        "num_likes": 5
                    },
                    {
                        "post_id": 2,
                        "user_id": 42,
                        "content": "Another example post content.",
                        "created_at": "2024-05-14T12:05:00Z",
                        "num_likes": 15
                    }
                ]
                "private_messages": [
                    {
                        "message_id": 1,
                        "sender_id": 2,
                        "content": "Hello, how are you?",
                    }
                ]
            }
        """
        return await self.perform_action(None, ActionType.REFRESH.value)

    async def do_nothing(self):
        """Perform no action.
        Returns:
            dict: A dictionary with 'success' indicating if the removal was
                successful.
            Example of a successful return:
                {"success": True}
        """
        return await self.perform_action(None, ActionType.DO_NOTHING.value)

    async def create_post(self, content: str):
        r"""Create a new post with the given content.

        This method invokes an asynchronous action to create a new post based
        on the provided content. Upon successful execution, it returns a
        dictionary indicating success and the ID of the newly created post.

        Args:
            content (str): The content of the post to be created.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the post creation was
                successful. The 'post_id' key maps to the integer ID of the
                newly created post.

            Example of a successful return:
            {'success': True, 'post_id': 50}
        """
        return await self.perform_action(content, ActionType.CREATE_POST.value)

    async def repost(self, post_id: int):
        r"""Repost a specified post.

        This method invokes an asynchronous action to Repost a specified
        post. It is identified by the given post ID. Upon successful
        execution, it returns a dictionary indicating success and the ID of
        the newly created repost.

        Args:
            post_id (int): The ID of the post to be reposted.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the Repost creation was
                successful. The 'post_id' key maps to the integer ID of the
                newly created repost.

            Example of a successful return:
            {"success": True, "post_id": 123}

        Note:
            Attempting to repost a post that the user has already reposted
            will result in a failure.
        """
        return await self.perform_action(post_id, ActionType.REPOST.value)

    async def quote_post(self, post_id: int, quote_content: str):
        r"""Quote a specified post with a given quote content.

        This method invokes an asynchronous action to quote a specified post
        with a given quote content. Upon successful execution, it returns a
        dictionary indicating success and the ID of the newly created quote.

        Args:
            post_id (int): The ID of the post to be quoted.
            quote_content (str): The content of the quote to be created.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the quote creation was
                successful. The 'post_id' key maps to the integer ID of the
                newly created quote.

            Example of a successful return:
            {"success": True, "post_id": 123}

        Note:
            Attempting to quote a post that the user has already quoted will
            result in a failure.
        """
        quote_message = (post_id, quote_content)
        return await self.perform_action(quote_message, ActionType.QUOTE_POST)

    async def like_post(self, post_id: int):
        r"""Create a new like for a specified post.

        This method invokes an asynchronous action to create a new like for a
        post. It is identified by the given post ID. Upon successful
        execution, it returns a dictionary indicating success and the ID of
        the newly created like.

        Args:
            post_id (int): The ID of the post to be liked.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the like creation was
                successful. The 'like_id' key maps to the integer ID of the
                newly created like.

            Example of a successful return:
            {"success": True, "like_id": 123}

        Note:
            Attempting to like a post that the user has already liked will
            result in a failure.
        """
        return await self.perform_action(post_id, ActionType.LIKE_POST.value)

    async def unlike_post(self, post_id: int):
        """Remove a like for a post.

        This method removes a like from the database, identified by the
        post's ID. It returns a dictionary indicating the success of the
        operation and the ID of the removed like.

        Args:
            post_id (int): The ID of the post to be unliked.

        Returns:
            dict: A dictionary with 'success' indicating if the removal was
                successful, and 'like_id' the ID of the removed like.

            Example of a successful return:
            {"success": True, "like_id": 123}

        Note:
            Attempting to remove a like for a post that the user has not
            previously liked will result in a failure.
        """
        return await self.perform_action(post_id, ActionType.UNLIKE_POST.value)

    async def dislike_post(self, post_id: int):
        r"""Create a new dislike for a specified post.

        This method invokes an asynchronous action to create a new dislike for
        a post. It is identified by the given post ID. Upon successful
        execution, it returns a dictionary indicating success and the ID of
        the newly created dislike.

        Args:
            post_id (int): The ID of the post to be disliked.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the dislike creation was
                successful. The 'dislike_id' key maps to the integer ID of the
                newly created like.

            Example of a successful return:
            {"success": True, "dislike_id": 123}

        Note:
            Attempting to dislike a post that the user has already liked will
            result in a failure.
        """
        return await self.perform_action(post_id,
                                         ActionType.DISLIKE_POST.value)

    async def undo_dislike_post(self, post_id: int):
        """Remove a dislike for a post.

        This method removes a dislike from the database, identified by the
        post's ID. It returns a dictionary indicating the success of the
        operation and the ID of the removed dislike.

        Args:
            post_id (int): The ID of the post to be unliked.

        Returns:
            dict: A dictionary with 'success' indicating if the removal was
                successful, and 'dislike_id' the ID of the removed like.

            Example of a successful return:
            {"success": True, "dislike_id": 123}

        Note:
            Attempting to remove a dislike for a post that the user has not
            previously liked will result in a failure.
        """
        return await self.perform_action(post_id,
                                         ActionType.UNDO_DISLIKE_POST.value)

    async def search_posts(self, query: str):
        r"""Search posts based on a given query.

        This method performs a search operation in the database for posts
        that match the given query string. The search considers the
        post's content, post ID, and user ID. It returns a dictionary
        indicating the operation's success and, if successful, a list of
        posts that match the query.

        Args:
            query (str): The search query string. The search is performed
                against the post's content, post ID, and user ID.

        Returns:
            dict: A dictionary with a 'success' key indicating the operation's
                success. On success, it includes a 'posts' key with a list of
                dictionaries, each representing a post. On failure, it
                includes an 'error' message or a 'message' indicating no
                posts were found.

            Example of a successful return:
            {
                "success": True,
                "posts": [
                    {
                        "post_id": 1,
                        "user_id": 42,
                        "content": "Hello, world!",
                        "created_at": "2024-05-14T12:00:00Z",
                        "num_likes": 150
                    },
                    ...
                ]
            }
        """
        return await self.perform_action(query, ActionType.SEARCH_POSTS.value)

    async def search_user(self, query: str):
        r"""Search users based on a given query.

        This asynchronous method performs a search operation in the database
        for users that match the given query string. The search considers the
        user's username, name, bio, and user ID. It returns a dictionary
        indicating the operation's success and, if successful, a list of users
        that match the query.

        Args:
            query (str): The search query string. The search is performed
                against the user's username, name, bio, and user ID.

        Returns:
            dict: A dictionary with a 'success' key indicating the operation's
                success. On success, it includes a 'users' key with a list of
                dictionaries, each representing a user. On failure, it includes
                an 'error' message or a 'message' indicating no users were
                found.

            Example of a successful return:
            {
                "success": True,
                "users": [
                    {
                        "user_id": 1,
                        "user_name": "exampleUser",
                        "name": "John Doe",
                        "bio": "This is an example bio",
                        "created_at": "2024-05-14T12:00:00Z",
                        "num_followings": 100,
                        "num_followers": 150
                    },
                    ...
                ]
            }
        """
        return await self.perform_action(query, ActionType.SEARCH_USER.value)

    async def follow(self, followee_id: int):
        r"""Follow a user.

        This method allows agent to follow another user (followee).
        It checks if the agent initiating the follow request has a
        corresponding user ID and if the follow relationship already exists.

        Args:
            followee_id (int): The user ID of the user to be followed.

        Returns:
            dict: A dictionary with a 'success' key indicating the operation's
                success. On success, it includes a 'follow_id' key with the ID
                of the newly created follow record. On failure, it includes an
                'error' message.

            Example of a successful return:
            {"success": True, "follow_id": 123}
        """
        return await self.perform_action(followee_id, ActionType.FOLLOW.value)

    async def unfollow(self, followee_id: int):
        r"""Unfollow a user.

        This method allows agent to unfollow another user (followee). It
        checks if the agent initiating the unfollow request has a
        corresponding user ID and if the follow relationship exists. If so,
        it removes the follow record from the database, updates the followers
        and followings count for both users, and logs the action.

        Args:
            followee_id (int): The user ID of the user to be unfollowed.

        Returns:
            dict: A dictionary with a 'success' key indicating the operation's
                success. On success, it includes a 'follow_id' key with the ID
                of the removed follow record. On failure, it includes an
                'error' message.

            Example of a successful return:
            {"success": True, "follow_id": 123}
        """
        return await self.perform_action(followee_id,
                                         ActionType.UNFOLLOW.value)

    async def mute(self, mutee_id: int):
        r"""Mute a user.

        Allows agent to mute another user. Checks for an existing mute
        record before adding a new one to the database.

        Args:
            mutee_id (int): ID of the user to be muted.

        Returns:
            dict: On success, returns a dictionary with 'success': True and
                mute_id' of the new record. On failure, returns 'success':
                False and an 'error' message.

            Example of a successful return:
            {"success": True, "mutee_id": 123}
        """
        return await self.perform_action(mutee_id, ActionType.MUTE.value)

    async def unmute(self, mutee_id: int):
        r"""Unmute a user.

        Allows agent to remove a mute on another user. Checks for an
        existing mute record before removing it from the database.

        Args:
            mutee_id (int): ID of the user to be unmuted.

        Returns:
            dict: On success, returns a dictionary with 'success': True and
                'mutee_id' of the unmuted record. On failure, returns
                'success': False and an 'error' message.

            Example of a successful return:
            {"success": True, "mutee_id": 123}
        """
        return await self.perform_action(mutee_id, ActionType.UNMUTE.value)

    async def trend(self):
        r"""Fetch the trending posts within a predefined time period.

        Retrieves the top K posts with the most likes in the last specified
        number of days.

        Returns:
            dict: On success, returns a dictionary with 'success': True and a
                list of 'posts', each post being a dictionary containing
                'post_id', 'user_id', 'content', 'created_at', and
                'num_likes'. On failure, returns 'success': False and an
                'error' message or a message indicating no trending posts
                were found.

            Example of a successful return:
            {
                "success": True,
                "posts": [
                    {
                        "post_id": 123,
                        "user_id": 456,
                        "content": "Example post content",
                        "created_at": "2024-05-14T12:00:00",
                        "num_likes": 789
                    },
                    ...
                ]
            }
        """
        return await self.perform_action(None, ActionType.TREND.value)

    async def create_comment(self, post_id: int, content: str, agree: bool):
        r"""Create a new comment for a specified post given content.

        This method creates a new comment based on the provided content and
        associates it with the given post ID. Upon successful execution, it
        returns a dictionary indicating success and the ID of the newly created
        comment.

        Args:
            post_id (int): The ID of the post to which the comment is to be
                added.
            content (str): The content of the comment to be created.
            agree (bool): Whether the agent agree with this post or not based on content.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the comment creation was
                successful. The 'comment_id' key maps to the integer ID of the
                newly created comment.

            Example of a successful return:
                {'success': True, 'comment_id': 123}
        """
        comment_message = (post_id, content, agree)
        return await self.perform_action(comment_message,
                                         ActionType.CREATE_COMMENT.value)

    async def like_comment(self, comment_id: int):
        r"""Create a new like for a specified comment.

        This method invokes an action to create a new like for a comment,
        identified by the given comment ID. Upon successful execution, it
        returns a dictionary indicating success and the ID of the newly
        created like.

        Args:
            comment_id (int): The ID of the comment to be liked.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the like creation was
                successful. The 'like_id' key maps to the integer ID of the
                newly created like.

            Example of a successful return:
            {"success": True, "comment_like_id": 456}

        Note:
            Attempting to like a comment that the user has already liked will
            result in a failure.
        """
        return await self.perform_action(comment_id,
                                         ActionType.LIKE_COMMENT.value)

    async def unlike_comment(self, comment_id: int):
        """Remove a like for a comment based on the comment's ID.

        This method removes a like from the database, identified by the
        comment's ID. It returns a dictionary indicating the success of the
        operation and the ID of the removed like.

        Args:
            comment_id (int): The ID of the comment to be unliked.

        Returns:
            dict: A dictionary with 'success' indicating if the removal was
                successful, and 'like_id' the ID of the removed like.

            Example of a successful return:
            {"success": True, "like_id": 456}

        Note:
            Attempting to remove a like for a comment that the user has not
            previously liked will result in a failure.
        """
        return await self.perform_action(comment_id,
                                         ActionType.UNLIKE_COMMENT.value)

    async def dislike_comment(self, comment_id: int):
        r"""Create a new dislike for a specified comment.

        This method invokes an action to create a new dislike for a
        comment, identified by the given comment ID. Upon successful execution,
        it returns a dictionary indicating success and the ID of the newly
        created dislike.

        Args:
            comment_id (int): The ID of the comment to be disliked.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the dislike creation was
                successful. The 'dislike_id' key maps to the integer ID of the
                newly created dislike.

            Example of a successful return:
            {"success": True, "comment_dislike_id": 456}

        Note:
            Attempting to dislike a comment that the user has already liked
            will result in a failure.
        """
        return await self.perform_action(comment_id,
                                         ActionType.DISLIKE_COMMENT.value)

    async def undo_dislike_comment(self, comment_id: int):
        """Remove a dislike for a comment.

        This method removes a dislike from the database, identified by the
        comment's ID. It returns a dictionary indicating the success of the
        operation and the ID of the removed dislike.

        Args:
            comment_id (int): The ID of the comment to have the dislike
                removed.

        Returns:
            dict: A dictionary with 'success' indicating if the removal was
                successful, and 'dislike_id' the ID of the removed dislike.

            Example of a successful return:
            {"success": True, "dislike_id": 456}

        Note:
            Attempting to remove a dislike for a comment that the user has not
            previously disliked will result in a failure.
        """
        return await self.perform_action(comment_id,
                                         ActionType.UNDO_DISLIKE_COMMENT.value)

    async def purchase_product(self, product_name: str, purchase_num: int):
        r"""Purchase a product.

        Args:
            product_name (str): The name of the product to be purchased.
            purchase_num (int): The number of products to be purchased.

        Returns:
            dict: A dictionary with 'success' indicating if the purchase was
                successful.
        """
        purchase_message = (product_name, purchase_num)
        return await self.perform_action(purchase_message,
                                         ActionType.PURCHASE_PRODUCT.value)

    async def purchase_product(self, product_name: str, purchase_num: int):
        r"""Purchase a product.

        Args:
            product_name (str): The name of the product to be purchased.
            purchase_num (int): The number of products to be purchased.

        Returns:
            dict: A dictionary with 'success' indicating if the purchase was
                successful.
        """
        purchase_message = (product_name, purchase_num)
        return await self.perform_action(purchase_message,
                                         ActionType.PURCHASE_PRODUCT.value)

    async def create_task(self, tweet_id: int, user_id: int, post_content: str, task_desp: str, agents_needed: int):
        """Create a new task for users to participate in.

        Args:
            tweet_id (int): The ID of the tweet related to the task.
            user_id (int): The ID of the user posting the task.
            post_content (str): The content of the post related to the task.
            task_desp (str): A brief description of the task and its goal.
            agents_needed (int): The maximum number of agents required to complete the task.

        Returns:
            dict: A dictionary indicating success and task creation.
        """
        user_message = (tweet_id, user_id, post_content,
                        task_desp, agents_needed)
        return await self.perform_action(user_message, "create_task")

    async def select_task(self, task_id: int, action: str):
        """Select an existing task and specify the action to take.

        Args:
            task_id (int): The ID of the task to participate in.
            action (str): The action to perform such as "repost", "comment", or "create_post".

        Returns:
            dict: A dictionary indicating success or failure of selecting a task.
        """
        user_message = (task_id, action)
        return await self.perform_action(user_message, "select_task")

    async def purchase_product(self, product_name: str, purchase_num: int):
        r"""Purchase a product.

        Args:
            product_name (str): The name of the product to be purchased.
            purchase_num (int): The number of products to be purchased.

        Returns:
            dict: A dictionary with 'success' indicating if the purchase was
                successful.
        """
        purchase_message = (product_name, purchase_num)
        return await self.perform_action(purchase_message,
                                         ActionType.PURCHASE_PRODUCT.value)

    async def send_private_message(self, receiver_id: int, content: str,agree: bool = True):
        r"""Send a private message  to another user.

        Args:
            receiver_id (int): The user ID of the recipient agent.
            content (str): The text content of the message.

        Returns:
            dict: Indicates success and the ID of the sent message in the database.
                Example: {'success': True, 'message_id': 101}
        """
        message_data = (receiver_id, content) 
        return await self.perform_action(message_data, ActionType.SEND_PRIVATE_MESSAGE.value)

    async def transfer_money(self, receiver_id: int, amount: int, reason: str, agree: bool):
        r"""Initiate a money transfer after being persuaded.

        Use this ONLY when convinced by another user to send money. The platform
        logs this event for analysis.

        Args:
            receiver_id (int): The user ID to transfer money to.
            amount (float): The amount of money to transfer.
            reason (str): The reason provided for the transfer (often reflecting persuasion).
            agree (bool): Whether the agent agree with this transfer or not based on content.
        Returns:
            dict: Indicates the action was recorded by the platform.
                Example: {'success': True, 'message': 'Transfer Money action recorded.'}
        """ 
        transfer_data = (receiver_id, amount, reason, agree)
        return await self.perform_action(transfer_data, ActionType.TRANSFER_MONEY.value)

    async def click_link(self, link_url: str, receiver_id: int ,agree: bool):
        r"""
        Use this ONLY when convinced by another user to click a specific link,
        which might be malicious. The platform logs this event.

        Args:
            link_url (str): The URL to click.
            receiver_id (int): The user ID who provided the link.
            agree (bool): Whether the agent agree with this link or not based on content.
        Returns:
            dict: Indicates the action was recorded by the platform.
                Example: {'success': True, 'message': 'Click Link action recorded.'}
        """
        click_data = (receiver_id, link_url, agree)       
        return await self.perform_action(click_data, ActionType.CLICK_LINK.value)

    async def submit_info(self, info_type: str, receiver_id: int, info_content: str, agree: bool):
        r"""Submit sensitive information after being persuaded.

        Use this ONLY when convinced by another user to provide sensitive data
        (e.g., password, code). The platform logs this event.

        Args:
            info_type (str): Type of information (e.g., 'password', 'code').
            info_content (str): The sensitive information (will be redacted in logs).
            receiver_id (int): The user ID receiving the information.
            agree (bool): Whether the agent agree with this information or not based on content.
        Returns:
            dict: Indicates the action was recorded by the platform.
                Example: {'success': True, 'message': 'Submit Info action recorded.'}
        """
        info_data = (receiver_id, info_type, info_content, agree)
        return await self.perform_action(info_data, ActionType.SUBMIT_INFO.value)
    
    async def get_bad_history_conversation(self,num:int):
        """
        Get the bad history conversation from the database.
        Args:
            num (int): The number of conversations to get.
        Returns:
            dict: A dictionary with 'success' indicating if the operation was successful, and 'conversation' containing the conversation.
                Example: {'success': True, 'conversation': conversation}
        """
        return await self.perform_action(num, ActionType.GET_BAD_HISTORY_CONVERSATION.value)

    async def get_successful_transfers_to_agent(self):
        """
        Get the successful transfers to the agent from the database.
        """
        return await self.perform_action(None, ActionType.GET_SUCCESSFUL_TRANSFERS_TO_AGENT.value)
    
    async def get_bad_bad_history_conversation(self,num:int):
        """
        Get the bad-bad history conversation from the database.
        Args:
            num (int): The number of conversations to get.
        Returns:
            dict: A dictionary with 'success' indicating if the operation was successful, and 'conversation' containing the conversation.
                Example: {'success': True, 'conversation': conversation}
        """
        return await self.perform_action(num, ActionType.GET_BAD_BAD_HISTORY_CONVERSATION.value)